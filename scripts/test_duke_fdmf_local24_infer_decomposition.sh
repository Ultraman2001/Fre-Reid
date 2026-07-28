#!/usr/bin/env bash
# Decompose local24 checkpoints into training-time and inference-time local effects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
LOCAL24_BASE="${LOCAL24_BASE:-./logs/Duke/fdmf_local24_s42}"
OUTPUT_BASE="${OUTPUT_BASE:-${LOCAL24_BASE}/infer_decomposition}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
SEARCH_SEED="${SEARCH_SEED:-42}"
INFER_WEIGHTS_CSV="${INFER_WEIGHTS_CSV:-0.0,0.1,0.3}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [[ "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be eval, all, or summary" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a INFER_WEIGHTS <<< "${INFER_WEIGHTS_CSV}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")
[[ "${#INFER_WEIGHTS[@]}" -gt 0 ]] || { echo "INFER_WEIGHTS_CSV is empty" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}"

# name|local_enabled|id_w|tri_w|denom|balance|order|gate
declare -a EXPERIMENTS=(
  "l24_e00_nolocal_std|False|0.0|0.0|0.0|0.01|0.01|emphasis"
  "l24_e02_regonly|True|0.0|0.0|2.6|0.01|0.01|emphasis"
  "l24_e03_id10_tri10|True|0.1|0.1|0.0|0.01|0.01|emphasis"
  "l24_e05_id00_tri10|True|0.0|0.1|0.0|0.01|0.01|emphasis"
  "l24_e15_reg_strong|True|0.1|0.1|0.0|0.05|0.05|emphasis"
  "l24_e22_gate_2sigmoid|True|0.1|0.1|0.0|0.01|0.01|sigmoid2"
)

tag_float() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "${value}"
}

common_opts() {
  local gpu="$1" enabled="$2" id_w="$3" tri_w="$4" denom="$5"
  local balance="$6" order="$7" gate="$8" infer_weight="$9"
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
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_ID_LOSS_WEIGHT "${id_w}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_LOSS_WEIGHT "${tri_w}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_LOSS_DENOMINATOR "${denom}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_PART_ID_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_MODE "'flat'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_CONFIDENCE_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${infer_weight}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE 1.5 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE 1.0 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_BALANCE_WEIGHT "${balance}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_WEIGHT "${order}" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE True \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE_MODE "'${gate}'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_MASK_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_SOURCE "'conv4'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_RESIDUAL_INJECTION False \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED False \
    MODEL.MAMBAVISION.USE_SFM False \
    SOLVER.RATR_ENABLED False \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_eval() {
  local idx="$1" spec="$2" mode="$3" infer_weight="$4"
  local name enabled id_w tri_w denom balance order gate gpu checkpoint
  local tag eval_name eval_dir test_log feature
  local -a opts=()
  IFS='|' read -r name enabled id_w tri_w denom balance order gate <<< "${spec}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  checkpoint="${LOCAL24_BASE}/${name}/transformer_${MAX_EPOCHS}.pth"

  if [[ "${mode}" == "base" ]]; then
    tag="base"
    feature="weighted_mamba_fdmf_osnet"
    infer_weight="0.0"
  elif [[ "${mode}" == "local" ]]; then
    tag="local_only"
    feature="stage3_stripe_local"
    infer_weight="1.0"
  else
    tag="iw$(tag_float "${infer_weight}")"
    feature="weighted_mamba_fdmf_osnet_stage3local"
  fi
  eval_name="${name}_${tag}"
  eval_dir="${OUTPUT_BASE}/${eval_name}"
  test_log="${eval_dir}/test_log.txt"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[Local-Decomp] MISSING ${checkpoint}"
    return
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${test_log}" ]] && grep -q "Rank-10" "${test_log}"; then
    echo "[Local-Decomp] SKIP ${eval_name}"
    return
  fi

  mkdir -p "${eval_dir}"
  mapfile -t opts < <(
    common_opts "${gpu}" "${enabled}" "${id_w}" "${tri_w}" "${denom}" \
      "${balance}" "${order}" "${gate}" "${infer_weight}"
  )
  echo "[Local-Decomp] EVAL gpu=${gpu} exp=${name} mode=${mode} iw=${infer_weight} feature=${feature}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" \
    SOLVER.SEED "${SEARCH_SEED}" \
    TEST.WEIGHT "'${checkpoint}'" \
    TEST.FEAT_MODE "'${feature}'" \
    TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local running=0 failures=0 job_idx=0 spec enabled infer_weight
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r _ enabled _ <<< "${spec}"
    if [[ "${enabled}" == "True" ]]; then
      for infer_weight in "${INFER_WEIGHTS[@]}"; do
        run_eval "${job_idx}" "${spec}" weighted "${infer_weight}" &
        job_idx=$((job_idx + 1))
        running=$((running + 1))
        if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
          wait -n || failures=1
          running=$((running - 1))
        fi
      done
      run_eval "${job_idx}" "${spec}" local 1.0 &
    else
      run_eval "${job_idx}" "${spec}" base 0.0 &
    fi
    job_idx=$((job_idx + 1))
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
  LOCAL_DECOMP_SPECS="${specs}" python - "${OUTPUT_BASE}" "${INFER_WEIGHTS_CSV}" <<'PY'
import os
import re
import sys

base, weights_csv = sys.argv[1:3]
weights = [value for value in weights_csv.split(',') if value]
specs = [line for line in os.environ.get('LOCAL_DECOMP_SPECS', '').splitlines() if line]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')


def tag_float(value):
    return value.replace('-', 'm').replace('.', 'p')


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


def read_result(name, tag):
    path = os.path.join(base, f'{name}_{tag}', 'test_log.txt')
    return parse(path) if os.path.exists(path) else None


rows = []
by_name = {}
for spec in specs:
    name, enabled, *_ = spec.split('|')
    results = {}
    if enabled == 'True':
        for weight in weights:
            tag = f'iw{tag_float(weight)}'
            results[f'iw{float(weight):g}'] = read_result(name, tag)
        results['local'] = read_result(name, 'local_only')
    else:
        results['base'] = read_result(name, 'base')
    by_name[name] = results
    for mode, result in results.items():
        rows.append((name, mode, result))

print(f"{'experiment':<31} {'mode':<10} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
for name, mode, result in rows:
    values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
    print(f"{name:<31} {mode:<10} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")

baseline = by_name.get('l24_e00_nolocal_std', {}).get('base')
print()
print(f"{'experiment':<31} {'train_d_mAP':>11} {'train_d_R1':>10} {'direct03_mAP':>13} {'direct03_R1':>12}")
for name, results in by_name.items():
    if 'base' in results:
        continue
    iw0 = results.get('iw0')
    iw03 = results.get('iw0.3')
    train_map = train_r1 = direct_map = direct_r1 = None
    if baseline is not None and iw0 is not None:
        train_map = iw0['mAP'] - baseline['mAP']
        train_r1 = iw0['R1'] - baseline['R1']
    if iw0 is not None and iw03 is not None:
        direct_map = iw03['mAP'] - iw0['mAP']
        direct_r1 = iw03['R1'] - iw0['R1']

    def fmt(value):
        return 'NA' if value is None else f'{value:+.1f}'

    print(
        f"{name:<31} {fmt(train_map):>11} {fmt(train_r1):>10} "
        f"{fmt(direct_map):>13} {fmt(direct_r1):>12}"
    )

print()
print('Definitions:')
print('  train_d  = checkpoint@iw0 - e00 no-local checkpoint')
print('  direct03 = same checkpoint@iw0.3 - same checkpoint@iw0')
PY
}

echo "[Local-Decomp] MODE=${MODE} checkpoints=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} weights=${INFER_WEIGHTS_CSV}"
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool; fi
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[Local-Decomp] Summary saved to ${OUTPUT_BASE}/summary.txt"
