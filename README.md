# RL2: Ray Less Reinforcement Learning

A concise library of reinforcement learning for large language models.

This is the right library for you if you are tired with complicated abstractions.
We deliver a clear implementation within 1K lines.
You can simply launch the training with `torchrun` as you do in supervised fine-tuning.

Despite the simplicity, you should be able to scale up to moderate-sized, *e.g.*, 32B, language models with

* Model partition via Fully Sharded Data Parallelism
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
The data should be in the following format

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

### Reward

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

## Acknowledgement

This project is built upon the basis of many remarkable projects, including but not limited to
* [DeepSpeedChat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) for the proposal of hybrid engine
* [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention) for the support of ZigZag ring attention
* [SGLang](https://github.com/sgl-project/sglang) for the support of async inference engine

We also thank [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [veRL](https://github.com/volcengine/verl) for their pioneering work.

## Citation
If you find this library useful, please cite in the follwing format
```
@misc{Tan2025RL2,
    author={Chenmien Tan and Simon Yu and Lanbo Lin and Ze Zhang and Yuanwu Xu and Chenhao Jiang and Tianyuan Yang and Sicong Xie and Guannan Zhang},
    title={RL2: Ray Less Reinforcement Learning},
    note={GitHub repository},
    howpublished={\url{https://github.com/ChenmienTan/RL2}},
    year={2025}
}
```