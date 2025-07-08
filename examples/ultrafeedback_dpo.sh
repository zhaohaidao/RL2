torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.dpo \
    data.path=Chenmien/UltraFeedback \
    data.max_length=1024 \
    actor.model_name=allenai/Llama-3.1-Tulu-3-8B-SFT \
    actor.max_length_per_device=4096 \
    trainer.project=UltraFeedback \
    trainer.experiment_name=tulu-3-8b