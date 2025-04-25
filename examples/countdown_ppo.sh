export CUDA_VISIBLE_DEVICES=2,3,4,5

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py \
    data.train_data_path=data/countdown_train.json \
    data.test_data_path=data/countdown_test.json \
    data.prompts_per_rollout=256 \
    data.responses_per_prompt=4 \
    actor.model_name=Qwen/Qwen2.5-3B \
    actor.max_length_per_device=2048 \
    actor.save_dir=ckpts/countdown_ppo \
    actor.save_freq=1000 \
    actor.rollout.max_response_length=1024 \
    actor.rollout.reward_fn_path=rewards/countdown.py \
    actor.kl.coef=0.001 \
    actor.kl.type=reward \
    actor.kl.estimator=k1 \
    adv.estimator=gae \
    trainer.project=TinyZero \
    trainer.experiment_name=qwen2.5-3b-countdown-ppo \
    trainer.n_epochs=15 \
    trainer.test_freq=5 \
    trainer.disable_wandb=false