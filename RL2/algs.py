import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

def all_reduce_with_grad(partial_values, device_mesh):

    values = partial_values.detach()
    dist.all_reduce(
        values,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return values + partial_values - partial_values.detach()

def compute_logsumexp_by_chunk(logits, chunk_size=1024):
    
    logsumexp = []
    for start in range(0, logits.shape[1], chunk_size):
        logsumexp.append(
            logits[:, start:start + chunk_size].logsumexp(-1)
        )
    return torch.cat(logsumexp, -1)

def compute_logps(logits, actions, device_mesh):

    # When using tensor parallelism, each device will only have a shard of 
    # logits. We firstly figure out which device are the action logits on.
    rank = device_mesh.get_local_rank()

    local_vocab_size = torch.LongTensor(
        [logits.shape[-1]]
    ).to(torch.cuda.current_device())
    vocab_sizes = [
        torch.zeros_like(local_vocab_size)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        vocab_sizes,
        local_vocab_size,
        group=device_mesh.get_group()
    )
    cu_vocab_sizes = torch.cumsum(
        torch.cat(
            [torch.zeros_like(local_vocab_size)] + vocab_sizes
        ), 0
    )
    action_logit_device = (
        actions < cu_vocab_sizes[1:].unsqueeze(-1)
    ).to(torch.float32).argmax(0)
    local_action_indices = torch.where(
        action_logit_device == rank
    )[0]
    local_actions = actions[:, local_action_indices] - cu_vocab_sizes[rank]
    action_logits = torch.zeros(
        actions.shape, device=torch.cuda.current_device()
    )
    action_logits[:, local_action_indices] = torch.gather(
        logits[:, local_action_indices],
        dim=-1,
        index=local_actions.unsqueeze(-1)
    ).squeeze(-1)
    action_logits = all_reduce_with_grad(action_logits, device_mesh)

    partial_logsumexp = compute_logsumexp_by_chunk(logits)
    logsumexps = [
        torch.zeros_like(partial_logsumexp)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        logsumexps,
        partial_logsumexp,
        group=device_mesh.get_group()
    )
    logsumexps[rank] = partial_logsumexp
    logsumexp = torch.cat([
        logsumexp.unsqueeze(-1) for logsumexp in logsumexps
    ], -1).logsumexp(-1)

    return action_logits - logsumexp

def sequence_all_reduce(batch, values, device_mesh):
    # When using sequence parallelism, tokens are distributed 
    # across multiple devices, while it may require the sum 
    # of logps of all tokens to compute the loss in DPO.
    # We firstly compute the sum of logps, despite that the 
    # sum is not involved in the computation graph.
    cu_seqlens = batch["cu_seqlens"]
    values = torch.stack([
        values[:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    return all_reduce_with_grad(values, device_mesh)

def compute_kl_term(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    estimator: str
) -> torch.Tensor:
    # The (ref_)logps of non-action tokens are zero (see `Actor.
    # forward`), so their corresponding kl_term will also be zero.

    logp_diffs = logps - ref_logps
    if estimator == "k1":
        return logp_diffs
    elif estimator == "k2":
        return logp_diffs.pow(2) / 2
    elif estimator == "k3":
        return logp_diffs + torch.exp(- logp_diffs) - 1
    else:
        raise NotImplementedError

def compute_gae(data_list, gamma, lamda):

    # extract rewards and values of action tokens
    rewards, values, action_mask = [], [], []
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        rewards.append(ex["rewards"][indices])
        values.append(ex["values"][indices])
        action_mask.append(ex["action_mask"][indices])
    # pad to identical length for efficient computation
    rewards = pad_sequence(rewards, True)
    values = pad_sequence(values, True)
    action_mask = pad_sequence(action_mask, True)
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = torch.cat((values[:, 1:], torch.zeros((values.shape[0], 1))), -1)
    deltas = rewards + gamma * next_values - values

    # A_t = \delta_t + \gamma * \lambda * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + values

    action_gaes = gaes[torch.where(action_mask)]
    gaes = (gaes - action_gaes.mean()) * action_mask / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    for ex, gae, ret in zip(data_list, gaes, returns):
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["returns"] = torch.zeros_like(ex["rewards"])
        indices = torch.where(ex["action_mask"])[0]
        ex["advantages"][indices] = gae[:len(indices)]
        ex["returns"][indices] = ret[:len(indices)]

def compute_baseline(data_list, responses_per_prompt):

    rewards = torch.FloatTensor(
        [ex["rewards"].sum() for ex in data_list]
    ).view(-1, responses_per_prompt)

    return rewards, rewards.mean(-1)

def compute_reinforce_adv(
    data_list, responses_per_prompt, norm_var: bool
):

    rewards, baseline = compute_baseline(data_list, responses_per_prompt)
    advantages = rewards - baseline.unsqueeze(-1)

    if norm_var:
        stds = rewards.std(-1)
        advantages /= (
            stds.unsqueeze(-1) + torch.finfo(advantages.dtype).eps
        )

    for ex, advantage in zip(data_list, advantages.flatten()):
        ex["advantages"] = advantage * ex["action_mask"]