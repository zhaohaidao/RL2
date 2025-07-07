torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.sft \
    data.path=Chenmien/LIMO \
    data.max_length=16384 \
    data.batch_size=32 \
    actor.model_name=Qwen/Qwen2.5-7B-Instruct \
    actor.use_liger_kernel=true \
    actor.sp_size=2 \
    actor.max_length_per_device=8192 \
    trainer.project=LIMO \
    trainer.experiment_name=qwen2.5-7b-inst \
    trainer.n_epochs=15