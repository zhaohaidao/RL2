export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=31

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-conf=timeout=36000 \
    main.py