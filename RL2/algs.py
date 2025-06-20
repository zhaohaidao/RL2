from collections import defaultdict
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

def compute_logsumexp_by_chunk(logits, chunk_size=1024):
    
    logsumexp = []
    for start in range(0, logits.shape[1], chunk_size):
        logsumexp.append(
            logits[:, start:start+chunk_size].logsumexp(-1)
        )
    return torch.cat(logsumexp, -1)

def sequence_all_reduce(batch, values, device_mesh):
    # When using sequence parallelism, tokens are distributed 
    # across multiple devices, while it may require the avg ( 
    # resp. sum) of logps of all tokens to compute the loss in
    # SFT (resp. DPO).

    # We firstly compute the sum of logps, despite that the 
    # sum is not involved in the computation graph.
    cu_seqlens = batch["cu_seqlens"]
    partial_values = torch.stack([
        values[:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    values = partial_values.detach()
    dist.all_reduce(
        values,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    # Then, we keep the sum unchanged and let it involve in the
    # computation graph of the corresponding device.
    # All SP ranks will share identical loss, while they will 
    # perform backpropagation on their respective tokens.
    return values + partial_values - partial_values.detach()

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

def compute_baseline(data_list):

    rewards = [ex["rewards"].sum() for ex in data_list]

    uid2rewards = defaultdict(list)
    for ex, reward in zip(data_list, rewards):
        uid2rewards[ex["uid"]].append(reward)

    uid2baseline = {
        k: (torch.stack(v).mean() if len(v) > 1 else v[0])
        for k, v in uid2rewards.items()
    }

    return rewards, uid2rewards, uid2baseline

def compute_reinforce_adv(data_list, norm_var: bool):

    rewards, uid2rewards, uid2baseline = compute_baseline(data_list)
    for ex, reward in zip(data_list, rewards):
        ex["advantages"] = (reward - uid2baseline[ex["uid"]]) * ex["action_mask"]

    if norm_var:
        uid2std = {
            k: (torch.stack(v).std() if len(v) > 1 else 1)
            for k, v in uid2rewards.items()
        }
        for ex in data_list:
            ex["advantages"] /= (
                uid2std[ex["uid"]] + torch.finfo(ex["advantages"].dtype).eps
            )