#!/bin/bash
set -ex
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
 

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done  

conda_cmd="source activate && conda activate python310_torch25_cuda"
echo ${conda_cmd}
eval ${conda_cmd} 


ROOT_DIR=$(realpath ../../../../../)
MEGATRON_PATH=${MEGATRON_PATH:-${ROOT_DIR}}
 
export PYTHONPATH=${PYTHONPATH}:${MEGATRON_PATH}/v5000_megatron:${MEGATRON_PATH}/v5000_megatron/KLX-Megatron:${MEGATRON_PATH}/v5000_megatron/toolkits/distributed_checkpoints_convertor/impl
echo "PYTHONPATH: ${PYTHONPATH}"
 
source_cmd="source ${MEGATRON_PATH}/v5000_megatron/zj_examples/xpu_env.sh"
echo ${source_cmd}
eval ${source_cmd} 
unset USE_FAST_BF16_FC


#export XMLIR_DIST_ASYNC_ISEND_IRECV=false
#export DISABLE_XPYTORCH=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
 
 
NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
 
TP=${TP:-8}
PP=${PP:-1}
 
MODEL_SIZE=${MODEL_SIZE:-72B} # NOTE: not used
LOAD_DIR=${LOAD_DIR:-/mnt/lhycpfs/lhy/model/Qwen2.5-${MODEL_SIZE}/}
HF_DIR=${HF_DIR:-/mnt/lhycpfs/lhy/model/Qwen2.5-${MODEL_SIZE}/}
SAVE_DIR=${SAVE_DIR:-/mnt/lhycpfs/lhy/ckpt2/}/Qwen2.5-${MODEL_SIZE}-mcore-TP-${TP}-PP-${PP}
MG2HF=false
USE_CUDA=false
PR=bf16
 
 
 
OTHER_ARGS=()
if [ ${MG2HF} = true ]; then
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${HF_DIR}
        --hf-dir ${HF_DIR}
        --mcore2hf
    )
    mkdir -p ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
else
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${LOAD_DIR}
    )
    mkdir -p ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
fi
 
if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(
        --use-gpu        
    )
else
    OTHER_ARGS+=(
        --distributed-backend gloo
    )
fi
 
if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(
        --fp16
    )
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(
        --bf16
    )
fi
 
if [ -z ${NUM_NODES} ]; then
    echo "Please Provide WORLD_SIZE"
    exit
fi
 
if [ -z ${NODE_RANK} ]; then
    echo "Please Provide RANK"
    exit
fi
 
if [ -z ${MASTER_ADDR} ]; then
    echo "Please Provide MASTER_ADDR"
    exit
fi
 
if [ -z ${MASTER_PORT} ]; then
    echo "Please Provide MASTER_PORT"
    exit
fi
 
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
 
GPT_MODEL_ARGS=(
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --add-qkv-bias
    --seq-length 1
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --position-embedding-type rope
    --group-query-attention
)
 
if [ $MODEL_SIZE = 0.5B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 24
        --hidden-size 896
        --ffn-hidden-size 4864
        --num-attention-heads 14
        --num-query-groups 2
        --max-position-embeddings 32768
        --padded-vocab-size 151936
        --norm-epsilon 1e-6
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
elif [ $MODEL_SIZE = 1.5B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 28
        --hidden-size 1536
        --ffn-hidden-size 8960
        --num-attention-heads 12
        --num-query-groups 2
        --max-position-embeddings 32768
        --padded-vocab-size 151936
        --norm-epsilon 1e-6
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
elif [ $MODEL_SIZE = 3B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 36
        --hidden-size 2048
        --ffn-hidden-size 11008
        --num-attention-heads 16
        --num-query-groups 2
        --max-position-embeddings 32768
        --padded-vocab-size 151936
        --norm-epsilon 1e-6
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
elif [ $MODEL_SIZE = 7B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 28
        --hidden-size 3584
        --ffn-hidden-size 18944
        --num-attention-heads 28
        --untie-embeddings-and-output-weights
        --num-query-groups 4
        --max-position-embeddings 131072
        --padded-vocab-size 152064
        --norm-epsilon 1e-6
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
elif [ $MODEL_SIZE = 14B ]; then 
    GPT_MODEL_ARGS+=(
        --num-layers 48
        --hidden-size 5120
        --ffn-hidden-size 13824
        --num-attention-heads 40
        --untie-embeddings-and-output-weights
        --num-query-groups 8
        --max-position-embeddings 131072
        --padded-vocab-size 152064
        --norm-epsilon 1e-5
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
elif [ $MODEL_SIZE = 32B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 64
        --hidden-size 5120
        --ffn-hidden-size 27648
        --num-attention-heads 40
        --untie-embeddings-and-output-weights
        --num-query-groups 8
        --max-position-embeddings 131072
        --padded-vocab-size 152064
        --norm-epsilon 1e-5
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
elif [ $MODEL_SIZE = 72B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 80
        --hidden-size 8192
        --ffn-hidden-size 29568
        --num-attention-heads 64
        --untie-embeddings-and-output-weights
        --num-query-groups 8
        --max-position-embeddings 131072
        --padded-vocab-size 152064
        --norm-epsilon 1e-5
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
        )
    fi
fi
 
TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)
 
EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
)
 
CONVERT_ARGS=(
    --model-type GPT 
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    --no-load-optim
    --no-load-rng
    --logging-level 1
)
 
cmd="torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/v5000_megatron/toolkits/distributed_checkpoints_convertor/impl/convert.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}" \
    # --target-ckpt-format torch
    # distributed_backend
 
echo $cmd
eval $cmd
