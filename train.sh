#!/bin/bash
set -e

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
echo $ROOT_DIR

WORLD_SIZE=1
TQ_GPU_NUM=8
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-9988}

MODEL_SIZE=8B
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=${SEQ_LEN:-32768}
PAD_LEN=${SEQ_LEN}
LR_DECAY_STYLE=cosine
WEIGHT_DECAY=0.1
EXTRA_VOCAB_SIZE=256
TP=${TP:-8}
PP=${PP:-1}
CP=1
AC=${AC:-full}
TASK=sft # pretrain/sft
DATASET_TYPE=raw # mmap/raw
SAVE_INTERVAL=50000
# PRETRAIN_CHECKPOINT_PATH=${ROOT_DIR}/../../../model/Meta-Llama-3-8B
PRETRAIN_CHECKPOINT_PATH=/nas/qianhao/models/llama3-8b-mcore-tp${TP}-pp${PP}
# PRETRAIN_CHECKPOINT_PATH=/mnt/v5000-megatron/v5000-megatron/ckpt/Meta-Llama-3-8B-mcore-TP-${TP}-PP-${PP}
DATASET_PATH=/nas/qianhao/data/sample_long_sft_32k_48M.json
VALID_DATASET_PATH=${DATASET_PATH}
if [[ -z ${OUTPUT_DIR} ]];then
    OUTPUT_BASEPATH=${ROOT_DIR}/output
else
    OUTPUT_BASEPATH=${OUTPUT_DIR}
fi
MP_SFT_PACKING=false
CPT_CONTINUE=false
ASYNC_SAVE=false
USE_VIRTUAL_PP=false
USE_SWA=false
USE_FP8=false
PR=${PR:-bf16}
DO=true
FL=true
SP=true
TE=true
OPTIMIZER_OFFLOAD=false
MOE=false
SAVE_CKPT=true
RMS_NORM_EPS=1e-5

MEGATRON_PATH=/workspace/Megatron-LM
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${TQ_GPU_NUM}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


if [ $USE_FP8 = true ]; then
    PR=fp8
fi


if echo "${DATASET_PATH}" | grep -q -E '\.txt$'
then
    DATASET_FILE=$DATASET_PATH
    DATASET_PATH="$(grep -v '^#' ${DATASET_FILE})"
    data_cache_options=" \
        --data-cache-path $OUTPUT_BASEPATH/data_cache"
else
    data_cache_options=" \
            "
fi
data_cache_options=" \
        --data-cache-path $OUTPUT_BASEPATH/llama3_8b/data_cache"


if [ $DATASET_TYPE = mmap ]; then
    dataset_type_options=" \
		    --dataset LLama-Pretrain-Idxmap \
            --data-path ${DATASET_PATH} \
            --split 99,1,0 "
elif [ $DATASET_TYPE = raw ]; then
    dataset_type_options=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type single \
        --dataset LLama-SFT-Raw "
fi

if [ $MODEL_SIZE = 0.5B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=896
NUM_ATTN_HEADS=14
INTERMEDIATE_SIZE=4864
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=131072
ROPE_THETA=500000
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

elif [ $MODEL_SIZE = 8B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=32768
ROPE_THETA=500000
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

elif [ $MODEL_SIZE = 70B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=28672
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
ROPE_THETA=500000
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

fi

if [ ${PP} -gt 1 ] && [ ${USE_VIRTUAL_PP} = true ]; then
    if [ $((NUM_LAYERS % PP)) -eq 0 ] && [ $((NUM_LAYERS / PP % 4)) -eq 0 ]; then
        VIRTUAL_PP=$((NUM_LAYERS / PP / 4))
        virtual_pp_options="--num-layers-per-virtual-pipeline-stage ${VIRTUAL_PP}"
    elif [ $((NUM_LAYERS % PP)) -eq 0 ] && [ $((NUM_LAYERS / PP % 2)) -eq 0 ]; then
        VIRTUAL_PP=$((NUM_LAYERS / PP / 2))
        virtual_pp_options="--num-layers-per-virtual-pipeline-stage ${VIRTUAL_PP}"
    else
        virtual_pp_options=""
    fi
else
    virtual_pp_options=""
fi

comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
            # --loss-scale 16384 \
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn \
		    --attention-backend flash"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi


if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi

if [ $MOE = true ]; then
    moe_options=" \
		    --moe-router-topk 1 \
		    --num-experts 8 \
		    --moe-aux-loss-coeff 1e-2 \
		    --expert-model-parallel-size 1 \
		    --moe-router-load-balancing-type aux_loss"

elif [ $MOE = false ]; then
    moe_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=" \
        --optimizer adam"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
        --reset-position-ids \
        --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

if [ ${USE_SWA} = true ]; then
    WINDOW_SIZE=$((SEQ_LEN / 8))
    swa_options=" \
        --window-size ${WINDOW_SIZE} 0 \
    "
else
    swa_options=""
fi

if [ -z ${ASYNC_SAVE} ]; then
    ASYNC_SAVE=false
fi

if [ ${ASYNC_SAVE} = true ]; then
    async_save_options=" \
        --async-save \
        --use-dist-ckpt
    "
else
    async_save_options=""
fi


if [ $TASK = pretrain ]; then
    task_options=" \
            --train-mode pretrain "
elif [ $TASK = sft ]; then
    task_options=" \
        --train-mode finetune \
        --eod-mask-loss "
fi


TRAIN_ITERS=100
LR_WARMUP_ITERS=0
# LR_DECAY_ITERS=${TRAIN_ITERS}

TASK_NAME="mcore-llama3-${MODEL_SIZE}-${TASK}"
DETAIL_TASK_NAME="${TASK_NAME}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-virtual_pp-${VIRTUAL_PP}-ac-${AC}-do-${DO}-sp-${SP}"
CURRENT_TIME=$(date +"%m-%d-%H:%M")

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${TASK_NAME}"
LOG_DIR=${OUTPUT_BASEPATH}/log_${DETAIL_TASK_NAME}_${CURRENT_TIME}
LOG_NAME="${NODE_RANK}.txt"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${DETAIL_TASK_NAME}_${CURRENT_TIME}"


mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
mkdir -p ${LOG_DIR}
mkdir -p ${TENSORBOARD_DIR}


if [ $SAVE_CKPT = true ]; then
    save_ckpt_options=" \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --ckpt-format torch "
fi

find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

if [ -z ${CPT_CONTINUE} ] || [ ${CPT_CONTINUE} = false ]; then
    cpt_continue_options="\
     --no-load-optim \
     --no-load-rng "
elif [ ${CPT_CONTINUE} = true ];  then
    PRETRAIN_CHECKPOINT_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH}
    cpt_continue_options="\
        --no-load-rng "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH \
            --auto-detect-ckpt-format"
fi

        # --train-samples ${TRAIN_SAMPLES} \
megatron_options="  \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style ${LR_DECAY_STYLE} \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay ${WEIGHT_DECAY} \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-warmup-fraction 0.02 \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 0 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type LLama3Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --use-mcore-models \
        --rotary-base ${ROPE_THETA} \
        --distributed-timeout-minutes 40 \
        --calculate-per-token-loss \
        --log-mfu \
        --mfu-base-value 312 \
        "

if [[ -z ${LOG_FILE} ]];then
  LOG_FILE=${LOG_DIR}/${LOG_NAME}
fi

run_cmd="torchrun $DISTRIBUTED_ARGS train.py
 ${megatron_options} \
 ${save_ckpt_options} \
 ${pr_options} \
 ${load_options} \
 ${te_options} \
 ${activation_checkpoint_options} \
 ${do_options} \
 ${flash_options} \
 ${async_save_options} \
 ${sp_options} \
 ${gqa_options} \
 ${moe_options} \
 ${dataset_type_options} \
 ${offload_option} \
 ${comm_overlap_option} \
 ${task_options} \
 ${packing_options} \
 ${cpt_continue_options} \
 ${data_cache_options} \
 ${virtual_pp_options} \
 ${swa_options} \
 2>&1 | tee ${LOG_FILE}
 "
echo ${run_cmd}
eval ${run_cmd}
set +x
