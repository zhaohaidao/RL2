torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
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