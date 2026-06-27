#!/usr/bin/env bash
# Run the first lightweight MambaVision + OSNet descriptor-fusion checks.
set -euo pipefail

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-2}"
CONFIG="${CONFIG:-configs/OCC_Duke/mambavision_tiny_osnet_concat_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/osnet_fusion_ablation}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-}"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "mamba_single_clean|False|0.0|0.0"
  "mamba_osnet_concat_w05|True|0.5|1.0"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name enabled osnet_weight fused_weight

  IFS='|' read -r name enabled osnet_weight fused_weight <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"
  local osnet_pretrain_opts=()

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi

  echo "[OSNetFusion] GPU=${gpu} EXP=${name} enabled=${enabled} osnet_w=${osnet_weight} fused_w=${fused_weight}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED "${enabled}" \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT "${osnet_weight}" \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT "${fused_weight}" \
    "${osnet_pretrain_opts[@]}" \
    OUTPUT_DIR "${output_dir}"
}

running=0
failures=0
for idx in "${!EXPERIMENTS[@]}"; do
  run_experiment "${idx}" "${EXPERIMENTS[$idx]}" &
  running=$((running + 1))

  if [ "${running}" -ge "${MAX_JOBS}" ]; then
    if ! wait -n; then
      failures=1
    fi
    running=$((running - 1))
  fi
done

while [ "${running}" -gt 0 ]; do
  if ! wait -n; then
    failures=1
  fi
  running=$((running - 1))
done

echo "[OSNetFusion] All experiments finished. Logs: ${OUTPUT_BASE}"

if [ "${failures}" -ne 0 ]; then
  echo "[OSNetFusion] One or more experiments failed."
  exit 1
fi
