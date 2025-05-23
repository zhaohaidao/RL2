torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    data.train_data_path=data/orz.json \
    data.test_data_path=data/olympiadbench.json \
    data.prompts_per_rollout=128 \
    data.responses_per_prompt=64 \
    actor.model_name=Qwen/Qwen2.5-7B \
    actor.max_length_per_device=16384 \
    actor.save_freq=32 \
    actor.rollout.max_response_length=8192 \
    actor.rollout.env_path=envs/math.py \
    actor.rollout.multi_thread_scoring=false \
    actor.kl.coef=1e-3 \
    actor.kl.type=reward \
    actor.kl.estimator=k1 \
    adv.estimator=reinforce \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-reinforce \
    trainer.test_freq=8