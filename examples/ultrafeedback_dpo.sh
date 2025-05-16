# You should firstly convert dataset HuggingFaceH4/ultrafeedback_binarized 
# into the JSON file with desired format

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m RL2.trainer.dpo \
    data.path=data/ultrafeedback.json \
    data.max_length=1024 \
    data.batch_size_per_device=4 \
    actor.model_name=allenai/Llama-3.1-Tulu-3-8B-SFT \
    trainer.project=UltraFeedback \
    trainer.experiment_name=tulu-3-8b