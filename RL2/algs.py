from typing import List, Dict
from collections import defaultdict
import torch
import torch.distributed as dist

def tokenize_messages(tokenizer, messages):

    states, actions, action_mask = [], [], []
    for idx, message in enumerate(messages):

        state = tokenizer.apply_chat_template(
            messages[:idx + 1],
            add_generation_prompt=idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant"
        )[len(states):]

        states.extend(state)
        actions.extend(
            state if message["role"] == "assistant"
            else len(state) * [0]
        )
        action_mask.extend(len(state) * [
            1 if message["role"] == "assistant" else 0
        ])

    return {
        "states": states[:-1],
        "actions": actions[1:],
        "action_mask": action_mask[1:],
        "position_ids": list(range(len(states) - 1))
    }

def compute_seq_and_avg_logps(
    batch,
    logps,
    device_mesh
):

    cu_seqlens = batch["cu_seqlens"]
    partial_logps = torch.stack([
        logps[:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    logps = partial_logps.detach()
    dist.all_reduce(
        logps,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    logps = logps + partial_logps - partial_logps.detach()

    actions = torch.stack([
        batch["action_mask"][:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    dist.all_reduce(
        actions,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    avg_logps = logps / (actions + torch.finfo(logps.dtype).eps)

    return logps, avg_logps

def compute_kl_term(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    kl_estimator: str
) -> torch.Tensor:
    # The (ref_)logps of non-action tokens are zero (see `Actor.
    # forward`), so their corresponding kl_term will also be zero.

    logp_diffs = logps - ref_logps
    if kl_estimator == "k1":
        return logp_diffs
    elif kl_estimator == "k2":
        return logp_diffs.pow(2) / 2
    elif kl_estimator == "k3":
        return logp_diffs + torch.exp(- logp_diffs) - 1
    else:
        raise NotImplementedError

def compute_gae(
    data_list: List[Dict[str, torch.Tensor]],
    gamma: float,
    lamda: float
):

    for ex in data_list:

        # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
        # if s_{t+1} is a terminal state, V(s_{t+1}) = 0
        next_values = torch.cat(
            (ex["values"][:, 1:], torch.FloatTensor([[0]])),
        dim=-1)
        delta = ex["rewards"] + gamma * next_values - ex["values"]

        # A_t = \delta_t + \gamma * \lambda * A_{t+1}
        # if s_{t+1} is a terminal state, A_{t+1} = 0
        gae, reversed_gaes = 0, []
        for t in reversed(range(delta.shape[-1])):
            gae = delta[0, t] + gamma * lamda * gae
            reversed_gaes.append(gae)
        gaes = reversed_gaes[::-1]

        ex["advantages"] = torch.FloatTensor([gaes]) * ex["action_mask"]
        ex["returns"] = ex["advantages"] + ex["values"]

def compute_reinforce_adv(
    data_list: List[Dict[str, torch.Tensor]],
    norm_var: bool
):

    rewards = [ex["rewards"].sum() for ex in data_list]

    uid2rewards = defaultdict(list)
    for ex, reward in zip(data_list, rewards):
        uid2rewards[ex["uid"]].append(reward)

    uid2baseline = {
        k: (torch.stack(v).mean() if len(v) > 1 else v[0])
        for k, v in uid2rewards.items()
    }
    for ex, reward in zip(data_list, rewards):
        ex["advantages"] = (reward - uid2baseline[ex["uid"]]) * ex["action_mask"]

    if norm_var:
        uid2std = {
            k: (torch.stack(v).std() if len(v) > 1 else 1)
            for k, v in uid2rewards.items()
        }
        for ex in data_list:
            ex["advantages"] /= (uid2std[ex["uid"]] + torch.finfo(ex["advantages"].dtype).eps)