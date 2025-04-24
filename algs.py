from typing import List, Dict
import torch

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
    responses_per_prompt: int,
    norm_var: bool
):

    rewards = torch.FloatTensor(
        [ex["rewards"].sum() for ex in data_list]
    ).view(-1, responses_per_prompt)
    baselines = rewards.mean(-1, keepdim=True)
    advantages = rewards - baselines
    if norm_var:
        stds = rewards.std(-1, keepdim=True)
        advantages /= (stds + torch.finfo(stds.dtype).eps)
    advantages = advantages.flatten()

    for ex, advantage in zip(data_list, advantages):
        ex["advantages"] = advantage * ex["action_mask"]