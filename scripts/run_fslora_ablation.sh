#!/usr/bin/env bash
# Run FSLoRA adapter ablations on top of the current PAM + PADE aug + OSBBM baseline.
set -euo pipefail

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-2}"
CONFIG="${CONFIG:-configs/OCC_Duke/mambavision_tiny_transreid_pam_padeaug_osbbm_fslora_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/fslora_ablation}"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "fslora_r32_n2_freeze|32|2|0.001|2.0|0.30|0.40|True"
  "fslora_r32_n2_unfreeze|32|2|0.001|2.0|0.30|0.40|False"
  "fslora_r16_n4_freeze|16|4|0.001|2.0|0.30|0.40|True"
  "fslora_r64_n1_freeze|64|1|0.001|2.0|0.30|0.40|True"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name rank experts gamma lr_factor low_cutoff high_cutoff freeze_base

  IFS='|' read -r name rank experts gamma lr_factor low_cutoff high_cutoff freeze_base <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"

  echo "[FSLoRA] GPU=${gpu} EXP=${name} rank=${rank} experts=${experts} gamma=${gamma} lr_factor=${lr_factor} cutoffs=(${low_cutoff},${high_cutoff}) freeze_base=${freeze_base}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.FSLORA.ENABLED True \
    MODEL.FSLORA.RANK "${rank}" \
    MODEL.FSLORA.NUM_EXPERTS "${experts}" \
    MODEL.FSLORA.INIT_GAMMA "${gamma}" \
    MODEL.FSLORA.FREQ_LOW_CUTOFF "${low_cutoff}" \
    MODEL.FSLORA.FREQ_HIGH_CUTOFF "${high_cutoff}" \
    MODEL.FSLORA.FREEZE_BASE "${freeze_base}" \
    SOLVER.FSLORA_LR_FACTOR "${lr_factor}" \
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

echo "[FSLoRA] All experiments finished. Logs: ${OUTPUT_BASE}"

if [ "${failures}" -ne 0 ]; then
  echo "[FSLoRA] One or more experiments failed."
  exit 1
fi
