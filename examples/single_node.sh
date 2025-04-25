export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py --config-path=./config --config-name=countdown_config.yaml