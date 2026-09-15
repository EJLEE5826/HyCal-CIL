#!/bin/bash
set -e

GPU_ID=${1:-0}
DATA_ROOT=${DATA_ROOT:-./data}
OUTPUT_DIR=${OUTPUT_DIR:-./results}

for SEED in 0 1 42 1993; do
    python experiment_runner.py \
        --cfg config/data_config.yaml \
        --few_shot_cfg "config/few_shot_config_crossimbal_${SEED}.yaml" \
        --data_root "${DATA_ROOT}" \
        --gpu "${GPU_ID}" \
        --model_provider openai \
        --clip_model ViT-B/16 \
        --fusion sum \
        --sim_metric weighted \
        --n_min_p 60 \
        --scale 5 \
        --seed "${SEED}" \
        --seed_val "${SEED}" \
        --output_dir "${OUTPUT_DIR}"
done
