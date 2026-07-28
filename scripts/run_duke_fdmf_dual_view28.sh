#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# This experiment family is intentionally independent from the legacy
# INPUT.PAM BA/CA/EA path. Each training sample produces exactly two inputs:
# one for MambaVision and one for OSNet.
MODE="${MODE:-all}"                 # train / eval / all / summary
STAGE="${STAGE:-all}"               # a / b / c / bc / all
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_dual_view28_s42}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SEARCH_SEED="${SEARCH_SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
SELECTION_FILE="${OUTPUT_BASE}/selection.env"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi
if [[ "${STAGE}" != "a" && "${STAGE}" != "b" && "${STAGE}" != "c" && "${STAGE}" != "bc" && "${STAGE}" != "all" ]]; then
  echo "STAGE must be a, b, c, bc, or all" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|stage|family|dual|mode|prob|direction|pid_balance|crop_prob|fcu_exchange|local|fdmf_bypass|legacy_re
declare -a STAGE_A=(
  "dv28_a00_legacy_best|A|control|False|shared|0.5|mamba_erased|False|0.0|True|True|False|0.5"
  "dv28_a01_shared_p50|A|baseline|True|shared|0.5|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a02_clean100|A|marginal|True|shared|0.0|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a03_shared_p25|A|marginal|True|shared|0.25|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a04_shared_p75|A|marginal|True|shared|0.75|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a05_diffmask_p50|A|correlation|True|diffmask|0.5|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a06_independent|A|correlation|True|independent|0.5|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a07_anticorr_random|A|correlation|True|anticorr|1.0|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a08_anticorr_pid22|A|correlation|True|anticorr|1.0|mamba_erased|True|0.0|True|True|False|0.0"
  "dv28_a09_fixed_CE|A|direction|True|fixed|1.0|osnet_erased|False|0.0|True|True|False|0.0"
  "dv28_a10_fixed_EC|A|direction|True|fixed|1.0|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a11_anchor_CE_p50|A|direction|True|anchor|0.5|osnet_erased|False|0.0|True|True|False|0.0"
  "dv28_a12_anchor_EC_p50|A|direction|True|anchor|0.5|mamba_erased|False|0.0|True|True|False|0.0"
  "dv28_a13_state_sample|A|state_ref|True|state_sample|0.5|mamba_erased|False|0.0|True|True|False|0.0"
)

filter_specs() {
  local -n source_ref=$1
  local -n target_ref=$2
  target_ref=()
  if [[ -z "${EXPERIMENT_FILTER}" ]]; then
    target_ref=("${source_ref[@]}")
    return
  fi
  local spec name pattern
  IFS=',' read -r -a FILTERS <<< "${EXPERIMENT_FILTER}"
  for spec in "${source_ref[@]}"; do
    name="${spec%%|*}"
    for pattern in "${FILTERS[@]}"; do
      if [[ "${name}" == *"${pattern}"* ]]; then
        target_ref+=("${spec}")
        break
      fi
    done
  done
}

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_STAGE SPEC_FAMILY SPEC_DUAL SPEC_MODE \
    SPEC_PROB SPEC_DIRECTION SPEC_PID_BALANCE SPEC_CROP_PROB \
    SPEC_FCU_EXCHANGE SPEC_LOCAL SPEC_FDMF_BYPASS SPEC_LEGACY_RE <<< "$1"
}

common_opts() {
  local gpu="$1" dual="$2" route_mode="$3" route_prob="$4"
  local direction="$5" pid_balance="$6" crop_prob="$7"
  local fcu_exchange="$8" local_enabled="$9" fdmf_bypass="${10}"
  local legacy_re="${11}"
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
    MODEL.OSNET_FUSION.FCU_EXCHANGE_ENABLED "${fcu_exchange}" \
    MODEL.OSNET_FUSION.FCU_STAGES "[2,3]" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'mamba_to_osnet'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_BYPASS "${fdmf_bypass}" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MAMBA_SCAN_MODE "'raster'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_LEARNABLE_DIRECTION_WEIGHTS False \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.COMPLEMENTARITY.MODE "'none'" \
    MODEL.OSNET_FUSION.PEER_COMPLEMENT.ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${local_enabled}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TYPE "'semantic_detail'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_ID_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_LOSS_DENOMINATOR 2.6 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_PART_ID_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_MODE "'flat'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_CONFIDENCE_MODE "'none'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_GUIDED_TRIPLET_MIX 0.75 \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_GUIDED_TRIPLET_SOURCE "'main'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_DETACH_PROMPT True \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_DETACH_DETAIL False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT 0.0 \
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
    INPUT.DUAL_VIEW.ENABLED "${dual}" \
    INPUT.DUAL_VIEW.MODE "'${route_mode}'" \
    INPUT.DUAL_VIEW.PROB "${route_prob}" \
    INPUT.DUAL_VIEW.DIRECTION "'${direction}'" \
    INPUT.DUAL_VIEW.PID_BALANCED "${pid_balance}" \
    INPUT.DUAL_VIEW.CROP_PROB "${crop_prob}" \
    INPUT.RE_PROB "${legacy_re}" \
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
  local idx="$1" spec="$2"
  read_spec "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${SPEC_NAME}"
  local feature='weighted_mamba_fdmf_osnet_stage3local'
  local -a opts=()
  [[ "${SPEC_LOCAL}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[DualView28] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(
    common_opts "${gpu}" "${SPEC_DUAL}" "${SPEC_MODE}" "${SPEC_PROB}" \
      "${SPEC_DIRECTION}" "${SPEC_PID_BALANCE}" "${SPEC_CROP_PROB}" \
      "${SPEC_FCU_EXCHANGE}" "${SPEC_LOCAL}" "${SPEC_FDMF_BYPASS}" \
      "${SPEC_LEGACY_RE}"
  )
  echo "[DualView28] TRAIN gpu=${gpu} exp=${SPEC_NAME} stage=${SPEC_STAGE} family=${SPEC_FAMILY} dual=${SPEC_DUAL} route=${SPEC_MODE}/${SPEC_PROB}/${SPEC_DIRECTION} pid=${SPEC_PID_BALANCE} crop=${SPEC_CROP_PROB} fcu=${SPEC_FCU_EXCHANGE} local=${SPEC_LOCAL} bypass=${SPEC_FDMF_BYPASS}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
    TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2"
  read_spec "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${OUTPUT_BASE}/${SPEC_NAME}/transformer_${MAX_EPOCHS}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${SPEC_NAME}"
  local feature='weighted_mamba_fdmf_osnet_stage3local'
  local -a opts=()
  [[ "${SPEC_LOCAL}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  if [[ ! -f "${weight}" ]]; then
    echo "[DualView28] MISSING ${weight}"
    return
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[DualView28] SKIP eval ${SPEC_NAME}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(
    common_opts "${gpu}" "${SPEC_DUAL}" "${SPEC_MODE}" "${SPEC_PROB}" \
      "${SPEC_DIRECTION}" "${SPEC_PID_BALANCE}" "${SPEC_CROP_PROB}" \
      "${SPEC_FCU_EXCHANGE}" "${SPEC_LOCAL}" "${SPEC_FDMF_BYPASS}" \
      "${SPEC_LEGACY_RE}"
  )
  echo "[DualView28] EVAL gpu=${gpu} exp=${SPEC_NAME} feature=${feature}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'${feature}'" TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local action="$1"
  shift
  local -a specs=("$@")
  local running=0 failures=0 idx
  for idx in "${!specs[@]}"; do
    if [[ "${action}" == "train" ]]; then
      run_train "${idx}" "${specs[$idx]}" &
    else
      run_eval "${idx}" "${specs[$idx]}" &
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

summarize_specs() {
  local output_path="$1"
  shift
  local specs
  specs="$(printf '%s\n' "$@")"
  DUAL_VIEW28_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${output_path}" <<'PY'
import os
import re
import sys

base, epoch, output_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
specs = [line for line in os.environ.get('DUAL_VIEW28_SPECS', '').splitlines() if line]
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

rows = []
for spec in specs:
    fields = spec.split('|')
    name, stage, family = fields[:3]
    result = parse(os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt'))
    if result is not None:
        rows.append((fields, result))

header = (
    f"{'experiment':<29} {'st':<2} {'family':<11} {'mode':<12} {'p':>5} "
    f"{'direction':<14} {'pid':<5} {'crop':>5} {'fcu':<5} {'local':<5} "
    f"{'bypass':<6} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}"
)
lines = [header]
for fields, result in rows:
    name, stage, family, dual, mode, prob, direction, pid, crop, fcu, local, bypass, _ = fields
    lines.append(
        f"{name:<29} {stage:<2} {family:<11} {mode:<12} {prob:>5} "
        f"{direction:<14} {pid:<5} {crop:>5} {fcu:<5} {local:<5} "
        f"{bypass:<6} {result['mAP']:>6.1f} {result['R1']:>6.1f} "
        f"{result['R5']:>6.1f} {result['R10']:>6.1f}"
    )
text = '\n'.join(lines)
print(text)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text + '\n')
PY
}

select_from_stage_a() {
  local specs
  specs="$(printf '%s\n' "${STAGE_A[@]}")"
  DUAL_VIEW28_A_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${SELECTION_FILE}" <<'PY'
import os
import re
import sys

base, epoch, output_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
specs = [line for line in os.environ.get('DUAL_VIEW28_A_SPECS', '').splitlines() if line]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')

def parse(name):
    path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
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
    return result if 'mAP' in result and 'R1' in result else None

records = {}
for spec in specs:
    fields = spec.split('|')
    result = parse(fields[0])
    if result:
        records[fields[0]] = (fields, result)

baseline = records.get('dv28_a01_shared_p50')
candidate_names = [
    name for name in records
    if name.startswith(('dv28_a05_', 'dv28_a06_', 'dv28_a07_', 'dv28_a08_',
                        'dv28_a09_', 'dv28_a10_', 'dv28_a11_', 'dv28_a12_',
                        'dv28_a13_'))
]
eligible = []
for name in candidate_names:
    fields, result = records[name]
    if baseline is None or result['R1'] >= baseline[1]['R1'] - 0.2:
        eligible.append((result['mAP'], result['R1'], name, fields))
if not eligible:
    eligible = [
        (records[name][1]['mAP'], records[name][1]['R1'], name, records[name][0])
        for name in candidate_names
    ]

if eligible:
    _, _, route_name, route = max(eligible)
else:
    route_name = 'fallback_anchor_EC'
    route = [
        route_name, 'A', 'fallback', 'True', 'anchor', '0.5',
        'mamba_erased', 'False', '0.0', 'True', 'True', 'False', '0.0'
    ]

direction_candidates = [
    records[name] for name in ('dv28_a11_anchor_CE_p50', 'dv28_a12_anchor_EC_p50')
    if name in records
]
if direction_candidates:
    direction_fields, _ = max(
        direction_candidates,
        key=lambda item: (item[1]['mAP'], item[1]['R1']),
    )
    best_direction = direction_fields[6]
else:
    best_direction = 'mamba_erased'

with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(f"BEST_ROUTE_NAME='{route_name}'\n")
    handle.write(f"BEST_ROUTE_MODE='{route[4]}'\n")
    handle.write(f"BEST_ROUTE_PROB='{route[5]}'\n")
    handle.write(f"BEST_ROUTE_DIRECTION='{route[6]}'\n")
    handle.write(f"BEST_ROUTE_PID_BALANCE='{route[7]}'\n")
    handle.write(f"BEST_DIRECTION='{best_direction}'\n")
print(
    f"[DualView28] selected R*={route_name} "
    f"{route[4]}/{route[5]}/{route[6]} pid={route[7]}, D*={best_direction}"
)
PY
}

load_selection() {
  if [[ -f "${SELECTION_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${SELECTION_FILE}"
  else
    echo "[DualView28] No Stage-A selection found; using planned anchor E/C fallback."
    BEST_ROUTE_NAME='fallback_anchor_EC'
    BEST_ROUTE_MODE='anchor'
    BEST_ROUTE_PROB='0.5'
    BEST_ROUTE_DIRECTION='mamba_erased'
    BEST_ROUTE_PID_BALANCE='False'
    BEST_DIRECTION='mamba_erased'
  fi
}

build_stage_bc() {
  local reverse_direction='osnet_erased'
  if [[ "${BEST_DIRECTION}" == "osnet_erased" ]]; then
    reverse_direction='mamba_erased'
  fi
  STAGE_B=(
    "dv28_b00_shared_noexchange|B|fcu|True|shared|0.5|mamba_erased|False|0.0|False|True|False|0.0"
    "dv28_b01_route_noexchange|B|fcu|True|${BEST_ROUTE_MODE}|${BEST_ROUTE_PROB}|${BEST_ROUTE_DIRECTION}|${BEST_ROUTE_PID_BALANCE}|0.0|False|True|False|0.0"
    "dv28_b02_shared_nolocal|B|local|True|shared|0.5|mamba_erased|False|0.0|True|False|False|0.0"
    "dv28_b03_route_nolocal|B|local|True|${BEST_ROUTE_MODE}|${BEST_ROUTE_PROB}|${BEST_ROUTE_DIRECTION}|${BEST_ROUTE_PID_BALANCE}|0.0|True|False|False|0.0"
    "dv28_b04_shared_fdmfbypass|B|fdmf|True|shared|0.5|mamba_erased|False|0.0|True|True|True|0.0"
    "dv28_b05_route_fdmfbypass|B|fdmf|True|${BEST_ROUTE_MODE}|${BEST_ROUTE_PROB}|${BEST_ROUTE_DIRECTION}|${BEST_ROUTE_PID_BALANCE}|0.0|True|True|True|0.0"
    "dv28_b06_shared_crop25|B|crop|True|shared|0.5|mamba_erased|False|0.25|True|True|False|0.0"
    "dv28_b07_route_crop25|B|crop|True|${BEST_ROUTE_MODE}|${BEST_ROUTE_PROB}|${BEST_ROUTE_DIRECTION}|${BEST_ROUTE_PID_BALANCE}|0.25|True|True|False|0.0"
  )
  STAGE_C=(
    "dv28_c00_bestdir_p25|C|curve|True|anchor|0.25|${BEST_DIRECTION}|False|0.0|True|True|False|0.0"
    "dv28_c01_shared_dose0125|C|dose|True|shared|0.125|mamba_erased|False|0.0|True|True|False|0.0"
    "dv28_c02_bestdir_p75|C|curve|True|anchor|0.75|${BEST_DIRECTION}|False|0.0|True|True|False|0.0"
    "dv28_c03_shared_dose0375|C|dose|True|shared|0.375|mamba_erased|False|0.0|True|True|False|0.0"
    "dv28_c04_reverse_p25|C|reverse|True|anchor|0.25|${reverse_direction}|False|0.0|True|True|False|0.0"
    "dv28_c05_reverse_p75|C|reverse|True|anchor|0.75|${reverse_direction}|False|0.0|True|True|False|0.0"
  )
}

run_selected_specs() {
  local label="$1"
  shift
  local -a source=("$@")
  local -a selected=()
  filter_specs source selected
  if [[ "${#selected[@]}" -eq 0 ]]; then
    echo "[DualView28] No ${label} experiments matched filter."
    return
  fi
  echo "[DualView28] ${label}: ${#selected[@]} experiments"
  if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
    run_pool train "${selected[@]}"
  fi
  if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
    run_pool eval "${selected[@]}"
  fi
}

echo "[DualView28] MODE=${MODE} STAGE=${STAGE} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEARCH_SEED}"

if [[ "${STAGE}" == "a" || "${STAGE}" == "all" ]]; then
  run_selected_specs "Stage-A" "${STAGE_A[@]}"
  summarize_specs "${OUTPUT_BASE}/summary_stage_a.txt" "${STAGE_A[@]}"
  select_from_stage_a
fi

if [[ "${STAGE}" == "b" || "${STAGE}" == "c" || "${STAGE}" == "bc" || "${STAGE}" == "all" ]]; then
  load_selection
  build_stage_bc
  if [[ "${STAGE}" == "b" || "${STAGE}" == "bc" || "${STAGE}" == "all" ]]; then
    run_selected_specs "Stage-B" "${STAGE_B[@]}"
  fi
  if [[ "${STAGE}" == "c" || "${STAGE}" == "bc" || "${STAGE}" == "all" ]]; then
    run_selected_specs "Stage-C" "${STAGE_C[@]}"
  fi
  summarize_specs "${OUTPUT_BASE}/summary.txt" \
    "${STAGE_A[@]}" "${STAGE_B[@]}" "${STAGE_C[@]}"
else
  summarize_specs "${OUTPUT_BASE}/summary.txt" "${STAGE_A[@]}"
fi

echo "[DualView28] Finished. Summary: ${OUTPUT_BASE}/summary.txt"
echo "[DualView28] Selection: ${SELECTION_FILE}"
