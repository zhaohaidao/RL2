# RL2: Ray Less Reinforcement Learning

A concise library of reinforcement learning for large language models.

This is the right library for you if you are tired with complicated abstractions.
We deliver a clear implementation within 1K lines.
You can simply launch the training with `torchrun` as you do in supervised fine-tuning.

Despite the simplicity, you should be able to scale up to moderate-sized, *e.g.*, 32B, language models with

* Model partition via Fully Sharded Data Parallelism (and Tensor Parallelism upcoming!)
* Efficient sequence parallelism via [ZigZag Ring Attention](https://github.com/zhuzilin/ring-flash-attention)
* Inference engine and KV cache partition via Tensor Parallelism

We also support

* Balanced sequence packing for higher throughput
* Multi-turn rollout with [SGLang](https://github.com/sgl-project/sglang) async inference engine

RL2 is a production-ready library! Check our wandb report on [LIMO](https://wandb.ai/chenmientan/LIMO_archive), [SkyworkRM](https://wandb.ai/chenmientan/SkyworkRM_archive), [UltraFeedback](https://wandb.ai/chenmientan/UltraFeedback_archive), [OpenReasonerZero](https://wandb.ai/chenmientan/OpenReasonerZero_archive), and [SearchR1](https://wandb.ai/chenmientan/SearchR1_archive).

## Getting Started


### Installation

```
git clone https://github.com/ChenmienTan/RL2.git
cd RL2
pip install -e .
```


### Data

Hugging Face dataset and various file types, including JSON, JSONL, CSV, Parquet, and Arrow, are accepted.
The data for PPO should be in the following format

```
[
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "answer": "Beijing"
    }
]
```
For SFT
```
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"},
            {"role": "assistant", "content": "Beijing."}
        ]
    }
]
```
For reward modeling and DPO
```
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "chosen": "Beijing.",
        "rejected": "Shanghai."
    }
]
```

### Rewards

The reward function should be in the follwing format.
Specify the path to the Python script including the function via `actor.rollout.env_path`.

```
def reward_fn(messages, answer):
    pred = parse_answer(messages[-1]["content"])
    return float(is_equivalent(pred, answer))
```

If a reward model is used, it should be served outside of the training framework, *e.g.*, using vLLM or SGLang, and be accessed in the reward function.

### Tools

RL2 supports multi-turn rollout with function calling.
In this case, you should set `rollout.max_turns > 1` and include function `interact` with the following format in the Python script including the reward function.
You should parse the called functions in past messages and return new messages including the results.
An empty list indicates no function is called.

```
def interact(messages):
    queries = parse_query(messages[-1]["content])
    results = [search(query) for query in queries]
    return [
        {"role": "tool", "content": result}
        for result in results
    ]
```

### Training

Use `torchrun` to launch the training. For example, for single node
```
torchrun \
    --nproc_per_node=<number of GPUs> \
    -m RL2.trainer.ppo \
    <args>
```
For multi nodes
```
torchrun \
    --nnodes=<number of nodes> \
    --node_rank=<rank of node> \
    --nproc_per_node=<number of GPUs on a node> \
    --master_addr=<address of master node> \
    --master_port=<port of master node> \
    -m RL2.trainer.ppo \
    <args>
```

## Hyper-Parameters

### Data

* `path`: Hugging Face name or local path of dataset.
* `max_length`: The maximum length of a sequence.
* `batch_size`: `batch_size` samples will be used for an update.
* `prompts_per_rollout`: `prompts_per_rollout` prompts will be used per rollout.
* `responses_per_prompt`: `responses_per_prompt` trajectories will be sampled for a prompt in rollout.

### Actor and Critic

* `model_name`: Hugging Face name or local path of model.
* `gradient_checkpointing`: Whether to enable gradient checkpointing.
* `ddp_size`: The number of model parameter copies.
When `ddp_size=1`, ZeRO stage 3 is applied; when `ddp_size` equals to total number of GPUs, ZeRO stage 2 is applied.
* `tp_size`: The model parameter will be sharded across `tp_size` GPUs.
* `sp_size`: The sequence will be sharded across `sp_size` GPUs.
Must be divisible by the total number of GPUs.
* `optimizer_dir`: The directory of optimizer state to be loaded.  
* `max_length_per_device`: The maximum length allowed for a single GPU at training.
The length of any sequence cannot exceed `sp_size * max_length_per_device`.
* `max_inference_length_per_device`: The maximum length allowed for a single GPU at inference.
* `update_per_rollout`: The model will be updated `update_for_rollout` times per rollout.
* `clip`: The clipping range of logp ratio or values.
* `lr`: The learning rate of optimizer.
* `weight_decay`: The coefficient of L2 regularization of optimizer.
* `max_grad_norm`: The norm of gradient will be clipped to `max_grad_norm` if it exceeds the value.
* `warmup_ratio`: The fraction of steps to warm up the optimizer.
* `freeze_steps`: The model will be freezed in the first `freeze_steps` steps.
Should only be enabled for actor when `adv.estimator=gae` to warmup critic.
* `offload_model`: Whether to offload model when not needed.
* `offload_optimizer`: Whether to offload optimizer when not needed.
Notice that the optimization step will still run on GPUs, which differs from Adam offloading.
* `save_dir`: The directory of checkpoints to be saved.
* `save_freq`: A checkpoint will be saved every `save_freq` steps.
Default to `None`, where only a single checkpoint will be saved when the training is finished.
* `save_optimizer`: Whether to save the optimizer.

### Rollout

* `tp_size`: The inference engine will be sharded across `tp_size` GPUs.
Must be divisible by the total number of GPUs.
* `gpu_memory_utilization`: The fraction of memory reserved for inference engine.
* `train_sampling_params`: The sampling parameters for rollout in training.
At least `temperature` and `max_new_tokens` should be indicated.
* `max_turns`: The inference engine will generate at most `max_turns` times in a trajectory.
Default to `1`, where the inference engine will only generate once and no function will be called.
* `env_path`: The path to the Python script containing function `reward_fn` and `interact` (if `max_turns > 1`).

### KL

* `coef`: The coefficient of KL regularization.
* `type`: If `reward`, the KL estimator will be added into the reward of each action; if `loss`, the KL estimator will be added into the loss of policy gradient.
Should be set to `loss` for GRPO.
* `reward_estimator`: The estimator used to compute KL reward.
When `type=loss`, this will affect the value logged by wandb.
* `loss_estimator`: The estimator used to compute KL loss.
Should be set to `k3` for GRPO.

### Adv

* `estimator`: If `gae`, generalized advantage estimator (and hence critic) will be used to estimate advantage; if `reinforce`, normalized reward will be used to estimate advantage.
We use token-level loss, so the RL algorithm will be [Dr. GRPO](https://arxiv.org/abs/2503.20783) if `norm_var=False`.
* `gamma`: The discount factor of future rewards.
Will only be effective when `estimator=gae`.
* `lamda`: The coefficient to tradeoff variance and bias of generalized advantage estimator.
Will only be effective when `estimator=gae`.
* `norm_var`: Whether to divide advantages by the standard error.
Will only be effective when `estimator=reinforce`.
Should be set to `True` for GRPO.

### Trainer

* `project`: The name of wandb project.
* `experiment_name`: The name of wandb experiment.
* `n_epochs`: The dataset will be iterated through `n_epochs` times.
* `disable_wandb`: Whether to disable wandb.

## Acknowledgement

This project is built upon the basis of many remarkable projects, including but not limited to
* [DeepSpeedChat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) for the proposal of hybrid engine
* [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention) for the support of ZigZag ring attention
* [SGLang](https://github.com/sgl-project/sglang) for the support of async inference engine

We also thank [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [veRL](https://github.com/volcengine/verl) for their pioneering work.

## Citation
If you find this library useful, please cite in the following format
```
@misc{Tan2025RL2,
    author={Chenmien Tan and Simon Yu and Lanbo Lin and Ze Zhang and Yuanwu Xu and Chenhao Jiang and Tianyuan Yang and Sicong Xie and Guannan Zhang},
    title={RL2: Ray Less Reinforcement Learning},
    note={GitHub repository},
    howpublished={\url{https://github.com/ChenmienTan/RL2}},
    year={2025}
}
```

## We are Hiring

We are [Accio](https://www.accio.com/), the world's first B2B AI sourcing engine.
Send us an [email](mailto:accio241112@gmail.com) if you are interested in opportunities in agent and reinforcement learning.
