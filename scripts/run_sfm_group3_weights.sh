#!/bin/bash
# 自动运行 SFM 组3（权重消融）实验
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

CONFIG="configs/DukeMTMC/mambavision_tiny_transreid.yml"
OUTPUT_BASE="./output"

mkdir -p "${OUTPUT_BASE}"

# Group 3A: 最终级权重
LAMBDAS=("0.5" "1.0" "2.0")
LAMBDA_TAGS=("05" "10" "20")
BASE_AUX="0.2"

for idx in "${!LAMBDAS[@]}"; do
  lambda_val="${LAMBDAS[$idx]}"
  tag="${LAMBDA_TAGS[$idx]}"
  echo "[Group3A] Running SFM_LAMBDA=${lambda_val}, SFM_LAMBDA_AUX=${BASE_AUX} -> ${OUTPUT_BASE}/g3a_lambda_${tag}"
  python train.py --config_file "${CONFIG}" \
    SOLVER.SFM_LAMBDA "${lambda_val}" \
    SOLVER.SFM_LAMBDA_AUX "${BASE_AUX}" \
    OUTPUT_DIR "${OUTPUT_BASE}/g3a_lambda_${tag}"
done

# Group 3B: 中间级权重
AUX_WEIGHTS=("0.0" "0.2" "0.5")
AUX_TAGS=("00" "02" "05")
BASE_LAMBDA="1.0"

for idx in "${!AUX_WEIGHTS[@]}"; do
  aux_val="${AUX_WEIGHTS[$idx]}"
  tag="${AUX_TAGS[$idx]}"
  echo "[Group3B] Running SFM_LAMBDA=${BASE_LAMBDA}, SFM_LAMBDA_AUX=${aux_val} -> ${OUTPUT_BASE}/g3b_aux_${tag}"
  python train.py --config_file "${CONFIG}" \
    SOLVER.SFM_LAMBDA "${BASE_LAMBDA}" \
    SOLVER.SFM_LAMBDA_AUX "${aux_val}" \
    OUTPUT_DIR "${OUTPUT_BASE}/g3b_aux_${tag}"
done
