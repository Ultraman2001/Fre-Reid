#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_scan4_s42}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEARCH_SEED="${SEARCH_SEED:-42}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|scan_mode|learnable_direction_weights
declare -a EXPERIMENTS=(
  "fs4_raster_s${SEARCH_SEED}|raster|False"
  "fs4_axial4_s${SEARCH_SEED}|axial4|False"
  "fs4_snake4_s${SEARCH_SEED}|snake4|False"
  "fs4_snake4_lw_s${SEARCH_SEED}|snake4|True"
)

if [[ -n "${EXPERIMENT_FILTER}" ]]; then
  declare -a FILTERED=()
  for spec in "${EXPERIMENTS[@]}"; do
    [[ "${spec%%|*}" == *"${EXPERIMENT_FILTER}"* ]] && FILTERED+=("${spec}")
  done
  EXPERIMENTS=("${FILTERED[@]}")
fi
[[ "${#EXPERIMENTS[@]}" -gt 0 ]] || { echo "No experiments matched" >&2; exit 2; }

common_opts() {
  local gpu="$1" scan_mode="$2" learnable="$3"
  local -a pretrain_opts=()
  if [[ -n "${OSNET_PRETRAIN}" ]]; then
    pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi
  printf '%s\n' \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
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
    MODEL.OSNET_FUSION.FDMF_MAMBA_SCAN_MODE "'${scan_mode}'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_LEARNABLE_DIRECTION_WEIGHTS "${learnable}" \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.COMPLEMENTARITY.MODE "'none'" \
    MODEL.OSNET_FUSION.PEER_COMPLEMENT.ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED True \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TYPE "'semantic_detail'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE 1.5 \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE True \
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

run_train() {
  local idx="$1" spec="$2" name scan_mode learnable
  IFS='|' read -r name scan_mode learnable <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" output_dir="${OUTPUT_BASE}/${name}"
  local -a opts=()
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[FScan4] SKIP train ${name}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${scan_mode}" "${learnable}")
  echo "[FScan4] TRAIN gpu=${gpu} exp=${name} scan=${scan_mode} learnable=${learnable}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
    TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet_stage3local'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2" name scan_mode learnable
  IFS='|' read -r name scan_mode learnable <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${OUTPUT_BASE}/${name}/transformer_${MAX_EPOCHS}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${name}"
  local -a opts=()
  [[ -f "${weight}" ]] || { echo "[FScan4] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[FScan4] SKIP eval ${name}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${scan_mode}" "${learnable}")
  echo "[FScan4] EVAL gpu=${gpu} exp=${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet_stage3local'" \
    TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local action="$1" running=0 failures=0 idx
  for idx in "${!EXPERIMENTS[@]}"; do
    if [[ "${action}" == "train" ]]; then
      run_train "${idx}" "${EXPERIMENTS[$idx]}" &
    else
      run_eval "${idx}" "${EXPERIMENTS[$idx]}" &
    fi
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
  local specs="$(printf '%s\n' "${EXPERIMENTS[@]}")"
  FSCAN4_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" <<'PY'
import os
import re
import sys

base, epoch = sys.argv[1], int(sys.argv[2])
specs = [line for line in os.environ.get('FSCAN4_SPECS', '').splitlines() if line]
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

print(f"{'experiment':<27} {'scan':<8} {'learn':<6} {'ep':>4} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
final = []
for spec in specs:
    name, scan_mode, learnable = spec.split('|')
    path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
    result = parse(path) if os.path.exists(path) else None
    values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
    print(f"{name:<27} {scan_mode:<8} {learnable:<6} {epoch:>4} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")
    if result is not None:
        final.append((result['mAP'], result['R1'], name))
if final:
    best_map = max(final, key=lambda row: (row[0], row[1]))
    best_r1 = max(final, key=lambda row: (row[1], row[0]))
    print(f"\nbest_mAP: {best_map[2]} mAP={best_map[0]:.1f} R1={best_map[1]:.1f}")
    print(f"best_R1 : {best_r1[2]} mAP={best_r1[0]:.1f} R1={best_r1[1]:.1f}")
PY
}

echo "[FScan4] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[FScan4] Summary saved to ${OUTPUT_BASE}/summary.txt"
