#!/bin/bash
#SBATCH --job-name=cpt
#SBATCH --output=logs/%A_cpt.log
#SBATCH --gres=gpu:0

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- Static Config ---------------------------------------------- #
# SERVER_HOST="<TTT server IP>"  # set to TTT server IP
ZMQ_PORT=5555

N_ARTICLES=200  # Number of articles to use, -1 for all

OUTPUT_DIR="knowledge-incorporation/results/cpt"
mkdir -p "${OUTPUT_DIR}"

# -------- Experiment Grid -------------------------------------------- #
# Columns (space-separated):
#   TAG  DATASET  K_COMP  LORA_RANK  LORA_ALPHA  LORA_DROPOUT  EPOCHS  LR  BS  GA
EXPERIMENTS=(
    "base_5c  knowledge-incorporation/data/synthetic_data/eval/base_val.json  5  32  64  0  3  1e-3  4  2"
)

# -------- Loop & Launch ---------------------------------------------- #
for EXP in "${EXPERIMENTS[@]}"; do
    read -r TAG DATASET K_COMP LORA_RANK LORA_ALPHA LORA_DROPOUT EPOCHS LR BS GA SPLIT_NEWLINES <<< "${EXP}"

    OUT_TAG="${TAG}_$((RANDOM))"
    LOG_FILE="logs/${SLURM_JOB_ID}_cpt_${OUT_TAG}.log"

    echo "CPT run: ${OUT_TAG}"

    python3 -u -m knowledge-incorporation.src.query.CPT \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}$" \
        --server_host "${SERVER_HOST}" \
        --zmq_port "${ZMQ_PORT}" \
        --k_completions "${K_COMP}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --finetune_epochs "${EPOCHS}" \
        --finetune_lr "${LR}" \
        --batch_size "${BS}" \
        --gradient_accumulation_steps "${GA}" \
        --n_articles "${N_ARTICLES}" \
        ${SPLIT_NEWLINES:+--split_newlines} \
        >> "${LOG_FILE}" 2>&1

done

echo "Job finished."
