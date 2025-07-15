#!/bin/bash
set -e

WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29400}"
GPU_COUNT=$(nvidia-smi -L | wc -l)
TQ_GPU_NUM="${TQ_GPU_NUM:-$GPU_COUNT}"
MODEL_CLASS="${MODEL_CLASS:-qwen2}"
TP="${TP:-4}"
PP="${PP:-1}"
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256

MODEL_DIR="${MODEL_DIR:-/nas/qianhao/models/Qwen2.5-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas/qianhao/models/mcore-qwen2.5-7b-tp${TP}-pp${PP}}"
HF_CKPT_DIR="${HF_CKPT_DIR:-none}"
USE_TQLM_SPECS="${USE_TQLM_SPECS:-0}"
CONVERT_OPTION="${CONVERT_OPTION:-hf2mg}"

CONFIG_PATH="configs/default/model_convert.py"

TASK_TYPE=${TASK_TYPE:-sft}
RANK=${RANK:-0}
#TASK_TYPE=$(echo "$task_type" | tr '[:upper:]' '[:lower:]')


# 检查环境变量是否存在，并检查默认配置文件是否存在
if [[ -z "$TASK_TYPE" ]]; then
    echo "错误：缺少必要的环境变量。请确保task_type已设置。"
    exit 1
fi

if [[ "$CONVERT_OPTION" == "hf2mg" ]]; then
    DIR_OPTIONS=" \
	    --hf_model_dir ${MODEL_DIR} \
	    --mg_model_dir ${OUTPUT_DIR}"
elif [[ "$CONVERT_OPTION" == "mg2hf" ]]; then
	if [[ -v ITERATION ]]; then
		echo $ITERATION >$MG_MODEL_DIR/latest_checkpointed_iteration.txt
	fi

    DIR_OPTIONS=" \
	    --hf_model_dir ${OUTPUT_DIR} \
	    --mg_model_dir ${MODEL_DIR} \
	    --hf_ckpt_dir ${HF_CKPT_DIR}"
else
	echo "CONVERT_OPTION only have two option: hf2mg/mg2hf, but got $CONVERT_OPTION"
fi
SPEC_OPTIONS=""
if [[ ${USE_TQLM_SPECS} == 1 ]]; then
	SPEC_OPTIONS=" \
		--use_tqlm_spec"
	PYTHONPATH=/workspace/Megatron-LM-0.11.0:$PYTHONPATH
else
	PYTHONPATH=/workspace/Megatron-LM:$PYTHONPATH
fi
echo $PYTHONPATH
PYTHONPATH=$(pwd):$PYTHONPATH \
	torchrun --nproc_per_node 1 --nnodes 1 \
	--master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${RANK} \
	convert_model.py \
	--config_path ${CONFIG_PATH} \
	--task_type ${TASK_TYPE} \
	--model_class ${MODEL_CLASS} \
	--convert_option ${CONVERT_OPTION} \
	--tp ${TP} \
	--pp ${PP} \
	${SPEC_OPTIONS} \
	${DIR_OPTIONS}
