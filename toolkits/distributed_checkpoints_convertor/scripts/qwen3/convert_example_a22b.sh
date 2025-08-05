CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
bash ${CURRENT_DIR}/run_A22B_16xH20.sh A22B /mnt/v5000-megatron/v5000-megatron/model/Qwen3-235B-A22B/ /mnt/geogpt-training/home/qianhao/models/megatron_ckpt/mcore_qwen3_a22b_t4_p8_e8 false true bf16 /mnt/v5000-megatron/v5000-megatron/model/Qwen3-235B-A22B/
