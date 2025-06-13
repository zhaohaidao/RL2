# RL2: Ray Less Reinforcement Learning

A concise library of reinforcement learning for large language models.

This is the right library for you if you are tired with complicated abstractions and wish to learn reinforcement learning for large language models or perform a quick test for your own algorithm.
We deliver a clear implementation (see `RL2.trainer.ppo.PPOTrainer.train`) within 1K lines.
You can simply launch the training with `torchrun` as you do in supervised fine-tuning (see `examples/orz_reinforce.sh`).

Despite the simplicity, you should be able to scale up to moderate-sized, *e.g.*, 32B, language models with

* Model partition via Fully Sharded Data Parallelism
* Sequence parallelism via [ZigZag Ring Attention](https://github.com/zhuzilin/ring-flash-attention)
* Inference engine and KV cache partition via Tensor Parallelism

We also support

* Balanced sequence packing for higher throughput
* Multi-turn rollout with [SGLang](https://github.com/sgl-project/sglang) async inference engine

RL2 is a production-ready library! Check our wandb report on [UltraFeedback](https://wandb.ai/chenmientan/UltraFeedback_archive).

## Getting Started

Install the library using following command.
```
git clone https://github.com/ChenmienTan/RL2.git
cd RL2
pip install -e .
```

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