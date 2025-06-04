torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    data.train_data_path=train@Chenmien/Countdown \
    data.test_data_path=test@Chenmien/Countdown \
    data.prompts_per_rollout=256 \
    data.responses_per_prompt=16 \
    actor.model_name=Qwen/Qwen2.5-3B \
    actor.max_length_per_device=2048 \
    actor.save_freq=1000 \
    actor.rollout.max_response_length=1024 \
    actor.rollout.env_path=envs/countdown.py \
    adv.estimator=gae \
    trainer.project=TinyZero \
    trainer.experiment_name=qwen2.5-3b-countdown-ppo \
    trainer.n_epochs=15 \
    trainer.test_freq=5