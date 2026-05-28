#!/bin/bash
# 自动运行 SFM 组4（结构深度）实验
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

CONFIG="configs/DukeMTMC/mambavision_tiny_transreid.yml"
OUTPUT_BASE="./output"

mkdir -p "${OUTPUT_BASE}"

# Step 4-1: 级联范围
DEPTHS_STEP1=('[0, 0, 1]' '[0, 1, 1]' '[1, 1, 1]')
TAGS_STEP1=("001" "011" "111")

for idx in "${!DEPTHS_STEP1[@]}"; do
  depth_cfg="${DEPTHS_STEP1[$idx]}"
  tag="${TAGS_STEP1[$idx]}"
  echo "[Group4-Step1] Running SFM_DEPTHS=${depth_cfg} -> ${OUTPUT_BASE}/g4_depth_${tag}"
  python train.py --config_file "${CONFIG}" \
    MODEL.MAMBAVISION.SFM_DEPTHS "${depth_cfg}" \
    OUTPUT_DIR "${OUTPUT_BASE}/g4_depth_${tag}"
done

# Step 4-2: 深度分配（可选，通过 RUN_STEP2=1 开启）
RUN_STEP2=${RUN_STEP2:-0}
if [[ "${RUN_STEP2}" == "1" ]]; then
  DEPTHS_STEP2=('[2, 2, 2]' '[1, 2, 3]' '[3, 2, 1]')
  TAGS_STEP2=("222" "123" "321")
  for idx in "${!DEPTHS_STEP2[@]}"; do
    depth_cfg="${DEPTHS_STEP2[$idx]}"
    tag="${TAGS_STEP2[$idx]}"
    echo "[Group4-Step2] Running SFM_DEPTHS=${depth_cfg} -> ${OUTPUT_BASE}/g4_depth_${tag}"
    python train.py --config_file "${CONFIG}" \
      MODEL.MAMBAVISION.SFM_DEPTHS "${depth_cfg}" \
      OUTPUT_DIR "${OUTPUT_BASE}/g4_depth_${tag}"
  done
fi
