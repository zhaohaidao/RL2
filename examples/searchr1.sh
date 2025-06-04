# You should firstly follow https://github.com/PeterGriffinJin/Search-R1 
# to download the index and corpus.

python -m sglang.launch_server \
    --model-path intfloat/e5-base-v2 \
    --is-embedding \
    --tp 4 \
    --mem-fraction-static 0.1 &

python envs/local_search_service.py \
    --model_name intfloat/e5-base-v2 \
    --index_path data/e5_Flat.index \
    --corpus_path data/wiki-18.jsonl \
    --top_k 3 &

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health) -ne 200 ]; do
    sleep 1
done

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health) -ne 200 ]; do
    sleep 1
done

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    data.train_data_path=data/nq_train.json \
    data.test_data_path=data/nq_test.json \
    data.prompts_per_rollout=256 \
    data.responses_per_prompt=5 \
    actor.model_name=Qwen/Qwen2.5-7B-Instruct \
    actor.max_length_per_device=4096 \
    actor.rollout.gpu_memory_utilization=0.4 \
    actor.rollout.n_turns=4 \
    actor.rollout.max_response_length=512 \
    actor.rollout.env_path=envs/searchr1.py \
    adv.estimator=reinforce \
    trainer.project=SearchR1 \
    trainer.experiment_name=qwen3-8b_reinforce \
    trainer.test_freq=32