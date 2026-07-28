#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_local_causal8_s3407}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEARCH_SEED="${SEARCH_SEED:-3407}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
TRAIN_LOCAL_INFER_WEIGHT="0.3"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|enabled|guided_mix|detach_prompt|detach_detail|balance|order|gate
declare -a EXPERIMENTS=(
  "lc8_e00_nolocal|control|False|0.0|False|False|0.01|0.01|emphasis"
  "lc8_e01_local_self|baseline|True|0.0|False|False|0.01|0.01|emphasis"
  "lc8_e02_g75_full|guided|True|0.75|False|False|0.01|0.01|emphasis"
  "lc8_e03_g75_detach_prompt|causal|True|0.75|True|False|0.01|0.01|emphasis"
  "lc8_e04_g75_detach_detail|causal|True|0.75|False|True|0.01|0.01|emphasis"
  "lc8_e05_g75_detach_both|causal|True|0.75|True|True|0.01|0.01|emphasis"
  "lc8_e06_g75_regstrong|synthesis|True|0.75|False|False|0.05|0.05|emphasis"
  "lc8_e07_g75_sigmoid2|synthesis|True|0.75|False|False|0.01|0.01|sigmoid2"
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
  local detach_prompt="$4" detach_detail="$5"
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
    MODEL.OSNET_FUSION.STAGE3_LOCAL_ID_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_PART_ID_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_MODE "'flat'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_CONFIDENCE_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_GUIDED_TRIPLET_MIX "${guided_mix}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_GUIDED_TRIPLET_SOURCE "'main'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_DETACH_PROMPT "${detach_prompt}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_DETACH_DETAIL "${detach_detail}" \
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

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_FAMILY SPEC_ENABLED SPEC_MIX \
    SPEC_DETACH_PROMPT SPEC_DETACH_DETAIL SPEC_BALANCE SPEC_ORDER SPEC_GATE <<< "$1"
}

run_train() {
  local idx="$1" spec="$2"
  read_spec "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${SPEC_NAME}"
  local feature='weighted_mamba_fdmf_osnet_stage3local'
  local -a opts=()
  [[ "${SPEC_ENABLED}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[LocalCausal8] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(
    common_opts "${gpu}" "${SPEC_ENABLED}" "${SPEC_MIX}" \
      "${SPEC_DETACH_PROMPT}" "${SPEC_DETACH_DETAIL}" \
      "${SPEC_BALANCE}" "${SPEC_ORDER}" "${SPEC_GATE}" "${TRAIN_LOCAL_INFER_WEIGHT}"
  )
  echo "[LocalCausal8] TRAIN gpu=${gpu} exp=${SPEC_NAME} family=${SPEC_FAMILY} mix=${SPEC_MIX} detach=${SPEC_DETACH_PROMPT}/${SPEC_DETACH_DETAIL} reg=${SPEC_BALANCE}/${SPEC_ORDER} gate=${SPEC_GATE}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
    TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
}

run_eval_mode() {
  local gpu="$1" weight="$2" name="$3" enabled="$4"
  local mix="$5" detach_prompt="$6" detach_detail="$7"
  local balance="$8" order="$9" gate="${10}" mode="${11}"
  local feature infer_weight eval_dir
  local -a opts=()

  if [[ "${enabled}" != "True" ]]; then
    [[ "${mode}" == "base" ]] || return
    feature='weighted_mamba_fdmf_osnet'
    infer_weight="${TRAIN_LOCAL_INFER_WEIGHT}"
  else
    case "${mode}" in
      iw0)
        feature='weighted_mamba_fdmf_osnet_stage3local'
        infer_weight='0.0'
        ;;
      iw0.1)
        feature='weighted_mamba_fdmf_osnet_stage3local'
        infer_weight='0.1'
        ;;
      iw0.3)
        feature='weighted_mamba_fdmf_osnet_stage3local'
        infer_weight='0.3'
        ;;
      local)
        feature='stage3_stripe_local'
        infer_weight='0.3'
        ;;
      *)
        echo "Unknown eval mode: ${mode}" >&2
        return 2
        ;;
    esac
  fi

  eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${mode}/${name}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[LocalCausal8] SKIP eval ${name} ${mode}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(
    common_opts "${gpu}" "${enabled}" "${mix}" \
      "${detach_prompt}" "${detach_detail}" \
      "${balance}" "${order}" "${gate}" "${infer_weight}"
  )
  echo "[LocalCausal8] EVAL gpu=${gpu} exp=${name} mode=${mode} iw=${infer_weight}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'${feature}'" TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
}

run_eval_bundle() {
  local idx="$1" spec="$2"
  read_spec "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${OUTPUT_BASE}/${SPEC_NAME}/transformer_${MAX_EPOCHS}.pth"
  [[ -f "${weight}" ]] || { echo "[LocalCausal8] MISSING ${weight}"; return; }

  if [[ "${SPEC_ENABLED}" == "True" ]]; then
    local mode
    for mode in iw0 iw0.1 iw0.3 local; do
      run_eval_mode "${gpu}" "${weight}" "${SPEC_NAME}" "${SPEC_ENABLED}" \
        "${SPEC_MIX}" "${SPEC_DETACH_PROMPT}" "${SPEC_DETACH_DETAIL}" \
        "${SPEC_BALANCE}" "${SPEC_ORDER}" "${SPEC_GATE}" "${mode}"
    done
  else
    run_eval_mode "${gpu}" "${weight}" "${SPEC_NAME}" "${SPEC_ENABLED}" \
      "${SPEC_MIX}" "${SPEC_DETACH_PROMPT}" "${SPEC_DETACH_DETAIL}" \
      "${SPEC_BALANCE}" "${SPEC_ORDER}" "${SPEC_GATE}" base
  fi
}

run_pool() {
  local action="$1" running=0 failures=0 idx
  for idx in "${!EXPERIMENTS[@]}"; do
    if [[ "${action}" == "train" ]]; then
      run_train "${idx}" "${EXPERIMENTS[$idx]}" &
    else
      run_eval_bundle "${idx}" "${EXPERIMENTS[$idx]}" &
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
  local specs
  specs="$(printf '%s\n' "${EXPERIMENTS[@]}")"
  LOCAL_CAUSAL8_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" <<'PY'
import os
import re
import sys

base, epoch = sys.argv[1], int(sys.argv[2])
specs = [line for line in os.environ.get('LOCAL_CAUSAL8_SPECS', '').splitlines() if line]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')

def parse(path):
    result = {}
    if not os.path.exists(path):
        return None
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

def result_for(mode, name):
    return parse(os.path.join(base, 'eval', f'ep{epoch}', mode, name, 'test_log.txt'))

rows = []
for spec in specs:
    name, family, enabled, mix, detach_p, detach_d, balance, order, gate = spec.split('|')
    if enabled == 'True':
        modes = ('iw0', 'iw0.1', 'iw0.3', 'local')
    else:
        modes = ('base',)
    for mode in modes:
        result = result_for(mode, name)
        if result is not None:
            rows.append((name, family, mix, detach_p, detach_d, balance, order, gate, mode, result))

print(
    f"{'experiment':<29} {'family':<10} {'mix':>4} {'detach':<5} "
    f"{'reg':<9} {'gate':<9} {'mode':<6} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}"
)
for name, family, mix, detach_p, detach_d, balance, order, gate, mode, result in rows:
    print(
        f"{name:<29} {family:<10} {mix:>4} "
        f"{detach_p[0]}/{detach_d[0]:<3} {balance}/{order:<4} {gate:<9} {mode:<6} "
        f"{result['mAP']:>6.1f} {result['R1']:>6.1f} {result['R5']:>6.1f} {result['R10']:>6.1f}"
    )

spec_by_name = {spec.split('|')[0]: spec.split('|') for spec in specs}
results = {(name, mode): result for name, _, _, _, _, _, _, _, mode, result in rows}
control_name = next(
    (name for name, fields in spec_by_name.items() if fields[2] != 'True'),
    None,
)
control = results.get((control_name, 'base')) if control_name else None

print("\nDecomposition:")
print(f"{'experiment':<29} {'train_d_mAP':>11} {'train_d_R1':>10} {'direct01_mAP':>13} {'direct03_mAP':>13} {'direct03_R1':>12}")
best = []
for name, fields in spec_by_name.items():
    if fields[2] != 'True':
        continue
    iw0 = results.get((name, 'iw0'))
    iw01 = results.get((name, 'iw0.1'))
    iw03 = results.get((name, 'iw0.3'))
    if iw0 is None:
        continue
    train_map = iw0['mAP'] - control['mAP'] if control else None
    train_r1 = iw0['R1'] - control['R1'] if control else None
    direct01 = iw01['mAP'] - iw0['mAP'] if iw01 else None
    direct03_map = iw03['mAP'] - iw0['mAP'] if iw03 else None
    direct03_r1 = iw03['R1'] - iw0['R1'] if iw03 else None
    fmt = lambda value: 'NA' if value is None else f"{value:+.1f}"
    print(
        f"{name:<29} {fmt(train_map):>11} {fmt(train_r1):>10} "
        f"{fmt(direct01):>13} {fmt(direct03_map):>13} {fmt(direct03_r1):>12}"
    )
    if iw03 is not None:
        best.append((iw03['mAP'], iw03['R1'], name))
if best:
    winner = max(best, key=lambda row: (row[0], row[1]))
    print(f"\nbest_iw0.3: {winner[2]} mAP={winner[0]:.1f} R1={winner[1]:.1f}")
PY
}

echo "[LocalCausal8] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEARCH_SEED}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[LocalCausal8] Summary saved to ${OUTPUT_BASE}/summary.txt"
