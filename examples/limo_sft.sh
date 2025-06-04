torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m RL2.trainer.sft \
    data.path=Chenmien/LIMO \
    data.max_length=16384 \
    data.batch_size=32 \
    actor.model_name=Qwen/Qwen2.5-7B-Instruct \
    actor.sp_size=2 \
    trainer.project=LIMO \
    trainer.experiment_name=qwen2.5-7b-inst \
    trainer.n_epochs=15