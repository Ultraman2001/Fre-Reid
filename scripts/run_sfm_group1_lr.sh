#!/bin/bash
# 自动运行 SFM 组1（学习率倍率）实验
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG="configs/DukeMTMC/mambavision_tiny_transreid.yml"
OUTPUT_BASE="./output"

mkdir -p "${OUTPUT_BASE}"

declare -a FACTORS=("0.5" "1.0" "2.0")
declare -a TAGS=("05" "10" "20")

for idx in "${!FACTORS[@]}"; do
  factor="${FACTORS[$idx]}"
  tag="${TAGS[$idx]}"
  echo "[Group1] Running SFM_LR_FACTOR=${factor} -> ${OUTPUT_BASE}/g1_lr_${tag}"
  python train.py --config_file "${CONFIG}" \
    SOLVER.SFM_LR_FACTOR "${factor}" \
    OUTPUT_DIR "${OUTPUT_BASE}/g1_lr_${tag}"
done
