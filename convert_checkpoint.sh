TP=${TP:-8}
PP=${PP:-1}
MODEL_TYPE=${MODEL_TYPE:-"llama3"} #choices;'llama2-7B', 'llama2-13B', 'llama2-70B', 'llama2-7Bf', 'llama2-13Bf', 'llama2-70Bf', 'llama3', 'mistral', 'yi-34B', 'qwen2.5'
HF_FORMAT_DIR=${HF_FORMAT_DIR:-"/nas/qianhao/models/Meta-Llama-3-8B/"}
MEGATRON_FORMAT_DIR=${MEGATRON_FORMAT_DIR:-"/nas/qianhao/models/llama3-8b-mcore-tp${TP}-pp${PP}/"}
TOKENIZER_MODEL=${HF_FORMAT_DIR}
MEGATRON_PATH=/workspace/Megatron-LM/
export PYTHONPATH=${MEGATRON_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
python ${MEGATRON_PATH}/tools/checkpoint/convert.py \
  --bf16 \
  --model-type GPT \
  --loader llama_mistral \
  --saver core \
  --target-tensor-parallel-size ${TP} \
  --target-pipeline-parallel-size ${PP} \
  --checkpoint-type hf \
  --load-dir ${HF_FORMAT_DIR} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --model-size ${MODEL_TYPE} \
  --make-vocab-size-divisible-by 16 \
  --max-queue-size 8
