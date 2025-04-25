export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=31

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-conf=timeout=36000 \
    main.py \
    data.train_data_path=data/orz.json \
    data.test_data_path=data/olympiadbench.json \
    data.prompts_per_rollout=128 \
    data.responses_per_prompt=64 \
    actor.model_name=Qwen/Qwen2.5-7B \
    actor.max_length_per_device=16384 \
    actor.save_freq=32 \
    actor.rollout.max_response_length=8192 \
    actor.rollout.reward_fn_path=rewards/math.py \
    adv.estimator=gae \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-ppo \
    trainer.test_freq=8