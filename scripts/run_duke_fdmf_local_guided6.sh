#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_local_guided6_s42}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEARCH_SEED="${SEARCH_SEED:-42}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
LOCAL_INFER_WEIGHT="0.3"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|local_enabled|guided_mix
declare -a EXPERIMENTS=(
  "glt_e00_nolocal|control|False|0.0"
  "glt_e01_local_m00|baseline|True|0.0"
  "glt_e02_local_m25|guided|True|0.25"
  "glt_e03_local_m50|guided|True|0.5"
  "glt_e04_local_m75|guided|True|0.75"
  "glt_e05_local_m100|guided|True|1.0"
)

if [[ -n "${EXPERIMENT_FILTER}" ]]; then
  declare -a FILTERED=()
  IFS=',' read -r -a FILTERS <<< "${EXPERIMENT_FILTER}"
  for spec in "${EXPERIMENTS[@]}"; do
    name="${spec%%|*}"
    for pattern in "${FILTERS[@]}"; do
      if [[ "${name}" == *"${pattern}"* ]]; then
        FILTERED+=("${spec}")
        break
      fi
    done
  done
  EXPERIMENTS=("${FILTERED[@]}")
fi
[[ "${#EXPERIMENTS[@]}" -gt 0 ]] || { echo "No experiments matched" >&2; exit 2; }

common_opts() {
  local gpu="$1" enabled="$2" guided_mix="$3"
  local -a pretrain_opts=()
  if [[ -n "${OSNET_PRETRAIN}" ]]; then
    pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi
  printf '%s\n' \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
    MODEL.OSNET_FUSION.OSNET_TYPE "'osnet_x1_0'" \
    MODEL.OSNET_FUSION.FUSION_TYPE "'fdmf'" \
    MODEL.OSNET_FUSION.FUSION_NORM "'none'" \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT 0.5 \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT 1.0 \
    MODEL.OSNET_FUSION.FCU_ENABLED True \
    MODEL.OSNET_FUSION.FCU_STAGES "[2,3]" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'mamba_to_osnet'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MAMBA_SCAN_MODE "'raster'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_LEARNABLE_DIRECTION_WEIGHTS False \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.COMPLEMENTARITY.MODE "'none'" \
    MODEL.OSNET_FUSION.PEER_COMPLEMENT.ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${enabled}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TYPE "'semantic_detail'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_ID_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_PART_ID_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_MODE "'flat'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_CONFIDENCE_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_GUIDED_TRIPLET_MIX "${guided_mix}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_GUIDED_TRIPLET_SOURCE "'main'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${LOCAL_INFER_WEIGHT}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE 1.5 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE 1.0 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_BALANCE_WEIGHT 0.01 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_WEIGHT 0.01 \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE True \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE_MODE "'emphasis'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_MASK_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_SOURCE "'conv4'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_RESIDUAL_INJECTION False \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED False \
    MODEL.MAMBAVISION.USE_SFM False \
    SOLVER.OSNET_LR_FACTOR 2.0 \
    SOLVER.OSNET_WEIGHT_DECAY 0.0005 \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS 0.0005 \
    SOLVER.OSNET_FUSION_LR_FACTOR 3.0 \
    SOLVER.RATR_ENABLED False \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.CHECKPOINT_PERIOD 40 \
    SOLVER.EVAL_PERIOD "${MAX_EPOCHS}" \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_one() {
  local action="$1" idx="$2" spec="$3"
  local name family enabled guided_mix
  IFS='|' read -r name family enabled guided_mix <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"
  local feature='weighted_mamba_fdmf_osnet_stage3local'
  local -a opts=()
  [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${guided_mix}")

  if [[ "${action}" == "train" ]]; then
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
      echo "[Guided6] SKIP train ${name}"
      return
    fi
    mkdir -p "${output_dir}"
    echo "[Guided6] TRAIN gpu=${gpu} exp=${name} mix=${guided_mix}"
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
      "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
      TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
    return
  fi

  local weight="${output_dir}/transformer_${MAX_EPOCHS}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${name}"
  [[ -f "${weight}" ]] || { echo "[Guided6] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[Guided6] SKIP eval ${name}"
    return
  fi
  mkdir -p "${eval_dir}"
  echo "[Guided6] EVAL gpu=${gpu} exp=${name} mix=${guided_mix} iw=${LOCAL_INFER_WEIGHT}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'${feature}'" TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local action="$1" running=0 failures=0 idx
  for idx in "${!EXPERIMENTS[@]}"; do
    run_one "${action}" "${idx}" "${EXPERIMENTS[$idx]}" &
    running=$((running + 1))
    if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
      wait -n || failures=1
      running=$((running - 1))
    fi
  done
  while [[ "${running}" -gt 0 ]]; do
    wait -n || failures=1
    running=$((running - 1))
  done
  return "${failures}"
}

summarize() {
  local specs
  specs="$(printf '%s\n' "${EXPERIMENTS[@]}")"
  GUIDED6_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" <<'PY'
import os
import re
import sys

base, epoch = sys.argv[1], int(sys.argv[2])
specs = [line for line in os.environ.get('GUIDED6_SPECS', '').splitlines() if line]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')

def parse(path):
    result = {}
    with open(path, encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            match = map_re.search(line)
            if match:
                result = {'mAP': float(match.group(1))}
                continue
            match = rank_re.search(line)
            if match and 'mAP' in result:
                result['R' + match.group(1)] = float(match.group(2))
    return result if all(key in result for key in ('mAP', 'R1', 'R5', 'R10')) else None

print(f"{'experiment':<25} {'family':<10} {'mix':>5} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
final = []
for spec in specs:
    name, family, enabled, mix = spec.split('|')
    path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
    result = parse(path) if os.path.exists(path) else None
    values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
    print(f"{name:<25} {family:<10} {mix:>5} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")
    if result is not None:
        final.append((result['mAP'], result['R1'], name))
if final:
    best = max(final, key=lambda row: (row[0], row[1]))
    print(f"\nbest: {best[2]} mAP={best[0]:.1f} R1={best[1]:.1f}")
PY
}

echo "[Guided6] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEARCH_SEED} iw=${LOCAL_INFER_WEIGHT}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[Guided6] Summary saved to ${OUTPUT_BASE}/summary.txt"
