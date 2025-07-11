# RL2: Ray Less Reinforcement Learning

A concise library of reinforcement learning for large language models.

This is the right library for you if you want to learn reinforcement learning for large language models or have a quick test for your own algorithm.
We deliver a clear implementation within 1K lines.


Despite the simplicity, you should be able to scale up to moderate-sized, *e.g.*, 72B, language models with

* Model partition via [Fully Sharded Data Parallelism](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) and [Tensor Parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
* Efficient sequence parallelism via [ZigZag Ring Attention](https://github.com/zhuzilin/ring-flash-attention)
* Inference engine and KV cache partition via Tensor Parallelism

We also support

* Balanced sequence packing for higher throughput
* Multi-turn rollout with [SGLang](https://github.com/sgl-project/sglang) async inference engine

RL2 is a production-ready library! Check our wandb report on [OpenThoughts](https://wandb.ai/chenmientan/OpenThoughts_archive), [SkyworkRM](https://wandb.ai/chenmientan/SkyworkRM_archive), [UltraFeedback](https://wandb.ai/chenmientan/UltraFeedback_archive), [OpenReasonerZero](https://wandb.ai/chenmientan/OpenReasonerZero_archive), and [SearchR1](https://wandb.ai/chenmientan/SearchR1_archive).

## Getting Started


### Installation

```
git clone https://github.com/ChenmienTan/RL2.git
cd RL2
pip install -e .
```


### Data

Hugging Face dataset and various file types, *i.e.*, JSON, JSONL, CSV, Parquet, and Arrow, are accepted.
The data for SFT should be in the following format
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
For RM and DPO
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
For PPO
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

For SFT, RM, and DPO, `batch_size` samples will be used for an update.
For PPO, `prompts_per_rollout` prompts will be used per rollout and `responses_per_prompt` trajectories will be sampled for a prompt.
These trajectories will be evenly used for `update_per_rollout` updates.

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

## Guide for Hyper-Parameters

### Model Partition

* By default, *i.e.*, `ddp_size=1, tp_size=1`, your model will be partitioned via ZeRO stage 3.
* `ddp_size` specifies the number of model parameter copies.
For example, if you set `ddp_size` to the number of GPUs, your model will be partitioned by ZeRO stage 2.
Larger `ddp_size` leads to higher memory consumption and lower communication cost.
* For large models, sole data parallelism can be memory consuming.
You may specify `tp_size > 1` to enable tensor parallelism for higher throughput. 


### Sequence Length

For SFT, RM, and DPO, `max_length` is used to truncate sequences.
Notice that in RM and DPO, the chosen and rejected sequences will be packed together, so the actual sequence length can be up to twice of `max_length`.
For PPO, `max_new_tokens` is used to truncate generations.
The length of any sequence cannot exceed `sp_size * tp_size * max_length_per_device`.

### Algorithm

The default RL algorithm is [Dr. GRPO](https://arxiv.org/abs/2503.20783).
Specify `adv.estimator=gae` to use PPO or `adv.norm_var=true` and `kl.reward_estimator=k3` to use GRPO.

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
