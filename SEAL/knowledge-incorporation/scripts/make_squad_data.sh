#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/%A_mkdt.log
#SBATCH --gres=gpu:1

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- User-editable ---------------------------------------------- #
MODEL_NAME="Qwen/Qwen2.5-7B"  # model to use for data generation. For evaluation, set to the model to be evaluated. For RL training, set to the (n-1)'th RL checkpoint.
PORT=8001
DATASET_IN="knowledge-incorporation/data/squad_train.json"
DATASET_OUT="knowledge-incorporation/data/synthetic_data/train/iter0_train.json"
# DATASET_IN="knowledge-incorporation/data/squad_val.json"
# DATASET_OUT="knowledge-incorporation/data/synthetic_data/eval/base_val.json"

NUM_ARTICLES=50              # how many articles (-1 for all)
START_ARTICLE=0              # start from this article number
K=5                          # number of completions to generate per question
TEMPERATURE=1.0
TOP_P=0.95
MAX_TOKENS=8192
# --------------------------------------------------------------------- #

VLLM_HOST=$(hostname -i)
VLLM_URL="http://${VLLM_HOST}:${PORT}"

echo "Starting vLLM on GPU 0 → ${VLLM_URL}"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --host "$VLLM_HOST" \
    --port $PORT \
    --max-model-len $((MAX_TOKENS + 2048)) \
    --trust-remote-code \
    > "logs/${SLURM_JOB_ID}_vllm_mktr.log" 2>&1 &
VLLM_PID=$!

# Wait for health-check
until curl --silent --fail ${VLLM_URL}/health >/dev/null; do sleep 3; done
echo "vLLM ready."

python3 -m knowledge-incorporation.src.data_generation.make_squad_data \
    --model "$MODEL_NAME" \
    --vllm_api_url "$VLLM_URL" \
    --dataset_in "$DATASET_IN" \
    --dataset_out "$DATASET_OUT" \
    --n "$NUM_ARTICLES" \
    --start "$START_ARTICLE" \
    --k "$K" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    ${INSTRUCT_MODEL:+--instruct_model} \
    --max_tokens "$MAX_TOKENS"

echo "Shutting down vLLM"
kill $VLLM_PID

echo "Job finished."
