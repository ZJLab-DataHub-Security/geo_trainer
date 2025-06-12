TP=${TP:-8}
PP=${PP:-1}
MODEL_TYPE=${MODEL_TYPE:-"llama3"} #choices;'llama2-7B', 'llama2-13B', 'llama2-70B', 'llama2-7Bf', 'llama2-13Bf', 'llama2-70Bf', 'llama3', 'mistral', 'yi-34B', 'qwen2.5'
HF_FORMAT_DIR=${HF_FORMAT_DIR:-"/mnt/zj-gpfs/model/llama/Meta-Llama-3-8B"}
MEGATRON_FORMAT_DIR=${MEGATRON_FORMAT_DIR:-"/mnt/zj-gpfs/home/qianhao/models/llama3-8b-mcore-tp${TP}-pp${PP}/"}
TOKENIZER_MODEL=${HF_FORMAT_DIR}
MEGATRON_PATH=/workspace/Megatron-LM/
export PYTHONPATH=${MEGATRON_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo $MODEL_TYPE
if [ $MODEL_TYPE = llama3 ]; then
  vocab_size=128256
elif [ $MODEL_TYPE = "qwen2.5-7B" ]; then
  vocab_size=152064
fi
python ${MEGATRON_PATH}/tools/checkpoint/convert.py \
  --bf16 \
  --model-type GPT \
  --loader llama_mistral \
  --saver mcore \
  --target-tensor-parallel-size ${TP} \
  --target-pipeline-parallel-size ${PP} \
  --checkpoint-type hf \
  --load-dir ${HF_FORMAT_DIR} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --model-size ${MODEL_TYPE} \
  --make-vocab-size-divisible-by 16 \
  --true-vocab-size ${vocab_size} \
  --max-queue-size 8
cp ${HF_FORMAT_DIR}/*.json ${MEGATRON_FORMAT_DIR}/
cp ${HF_FORMAT_DIR}/*.txt ${MEGATRON_FORMAT_DIR}/
