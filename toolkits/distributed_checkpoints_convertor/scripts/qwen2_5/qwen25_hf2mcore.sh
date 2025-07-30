export MODEL_SIZE=7B
export LOAD_DIR=/mnt/v5000-megatron/v5000-megatron/model/Qwen2.5-7B
export SAVE_DIR=/mnt/v5000-megatron/v5000-megatron/ckpt2
TP=1 PP=1 bash run_hf2mcore_qwen2.5.sh
TP=2 PP=1 bash run_hf2mcore_qwen2.5.sh
TP=4 PP=1 bash run_hf2mcore_qwen2.5.sh

TP=1 PP=2 bash run_hf2mcore_qwen2.5.sh
TP=1 PP=4 bash run_hf2mcore_qwen2.5.sh

TP=2 PP=2 bash run_hf2mcore_qwen2.5.sh
TP=2 PP=4 bash run_hf2mcore_qwen2.5.sh


export MODEL_SIZE=72B
export LOAD_DIR=/mnt/v5000-megatron/v5000-megatron/model/Qwen2.5-72B
export SAVE_DIR=/mnt/v5000-megatron/v5000-megatron/ckpt2
TP=8 PP=1 bash run_hf2mcore_qwen2.5.sh
TP=4 PP=2 bash run_hf2mcore_qwen2.5.sh
TP=4 PP=1 bash run_hf2mcore_qwen2.5.sh
