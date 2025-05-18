# RL2: Ray Less Reinforcement Learning

A consise (~1K lines) implementation for hybrid engine without complicated abstraction, where all models, i.e., actor, inference engine, reference model, and critic, are colocated to eliminate GPU idle.
Due to the SPMD design and absence of parallel computing, the training is launched using `torchrun` rather than `ray`.

* Scalability: model partition by ZeRO (FSDP for flexible gradient accumulation), inference engine partition by TP, and sequence parallelism by Ring Attention (ZigZag for load balance).

* Efficiency: sequence packing for high throughput.

## Upcoming Features

- [ ] Reproduction wandb report for popular RL projects, e.g., OpenReasonerZero and TinyZero
- [ ] Multi-turn rollout with function calling
- [ ] Prefix caching in training

## Acknowledgement

This project cannot be done without [DeepSpeedChat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [veRL](https://github.com/volcengine/verl), and [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention).

## Citation
```
@misc{Tan2025RL2,
    author={Chenmien Tan and Simon Yu and Lanbo Lin and Ze Zhang and Yuanwu Xu and Chenhao Jiang and Tianyuan Yang and Sicong Xie and Guannan Zhang},
    title={RL2: Ray Less Reinforcement Learning},
    note={GitHub repository},
    howpublished={\url{https://github.com/ChenmienTan/RL2}},
    year={2025}
}
```
