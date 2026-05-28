#!/bin/bash
# 自动运行 SFM 组2（DropPath 正则）实验
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG="configs/DukeMTMC/mambavision_tiny_transreid.yml"
OUTPUT_BASE="./output"

mkdir -p "${OUTPUT_BASE}"

declare -a DROPS=("0.0" "0.05" "0.1" "0.2")
declare -a TAGS=("00" "005" "01" "02")

for idx in "${!DROPS[@]}"; do
  drop_val="${DROPS[$idx]}"
  tag="${TAGS[$idx]}"
  echo "[Group2] Running SFM_DROP_PATH=${drop_val} -> ${OUTPUT_BASE}/g2_drop_${tag}"
  python train.py --config_file "${CONFIG}" \
    MODEL.MAMBAVISION.SFM_DROP_PATH "${drop_val}" \
    OUTPUT_DIR "${OUTPUT_BASE}/g2_drop_${tag}"
done
