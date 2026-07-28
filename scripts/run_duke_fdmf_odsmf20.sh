#!/usr/bin/env bash
# ODSMF internal ablation on DukeMTMC-reID. The historical projection-Mamba
# baseline is intentionally not retrained in this sweep.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Usage: bash scripts/run_duke_fdmf_odsmf20.sh 0,1 4
# Four workers on two GPUs means two simultaneous jobs per GPU. Each fixed
# worker starts its next assigned experiment as soon as its current job ends.
MODE="${MODE:-all}"  # train / eval / all / summary
EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-odsmf20}"  # odsmf20 / purestate8
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0,1}}"
MAX_JOBS="${2:-4}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
if [[ "${EXPERIMENT_SUITE}" == "purestate8" ]]; then
  DEFAULT_OUTPUT_BASE="./logs/Duke/fdmf_purestate8_s42"
  RUN_TAG="PURESTATE8"
else
  DEFAULT_OUTPUT_BASE="./logs/Duke/fdmf_odsmf20_s42"
  RUN_TAG="ODSMF20"
fi
OUTPUT_BASE="${OUTPUT_BASE:-${DEFAULT_OUTPUT_BASE}}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
SEED="${SEED:-42}"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac
case "${EXPERIMENT_SUITE}" in
  odsmf20|purestate8) ;;
  *) echo "EXPERIMENT_SUITE must be odsmf20 or purestate8" >&2; exit 2 ;;
esac
if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|C|D|share_mamba|share_norm|dim|depth|packed|basis|gate|shuffle_D|detach_D|carrier|alpha
declare -a ODSMF20_EXPERIMENTS=(
  "od20_e00_dual_shared|anchor|True|True|True|False|256|1|False|fixed|separable|False|False|True|0.1"
  "od20_e01_consensus_only|state|True|False|True|False|256|1|False|fixed|separable|False|False|True|0.1"
  "od20_e02_discrepancy_only|state|False|True|True|False|256|1|False|fixed|separable|False|False|True|0.1"
  "od20_e03_dual_independent|sharing|True|True|False|False|256|1|False|fixed|separable|False|False|True|0.1"
  "od20_e04_dual_tied_norm|normalization|True|True|True|True|256|1|False|fixed|separable|False|False|True|0.1"
  "od20_e05_width128|width|True|True|True|False|128|1|False|fixed|separable|False|False|True|0.1"
  "od20_e06_width512|width|True|True|True|False|512|1|False|fixed|separable|False|False|True|0.1"
  "od20_e07_depth2|depth|True|True|True|False|256|2|False|fixed|separable|False|False|True|0.1"
  "od20_e08_packed_cd_scan|topology|True|True|True|False|256|1|True|fixed|separable|False|False|True|0.1"
  "od20_e09_absdiff|basis|True|True|True|False|256|1|False|absdiff|separable|False|False|True|0.1"
  "od20_e10_signed_abs|basis|True|True|True|False|256|1|False|signed_abs|separable|False|False|True|0.1"
  "od20_e11_learned_orthogonal|basis|True|True|True|False|256|1|False|learned_orthogonal|separable|False|False|True|0.1"
  "od20_e12_learned_free|basis|True|True|True|False|256|1|False|learned_free|separable|False|False|True|0.1"
  "od20_e13_no_gate|gate|True|True|True|False|256|1|False|fixed|none|False|False|True|0.1"
  "od20_e14_spatial_gate|gate|True|True|True|False|256|1|False|fixed|spatial|False|False|True|0.1"
  "od20_e15_channel_gate|gate|True|True|True|False|256|1|False|fixed|channel|False|False|True|0.1"
  "od20_e16_shuffle_discrepancy|causal|True|True|True|False|256|1|False|fixed|separable|True|False|True|0.1"
  "od20_e17_detach_discrepancy|gradient|True|True|True|False|256|1|False|fixed|separable|False|True|True|0.1"
  "od20_e18_pure_dualstate|carrier|True|True|True|False|256|1|False|fixed|separable|False|False|False|0.1"
  "od20_e19_alpha_zero|initialization|True|True|True|False|256|1|False|fixed|separable|False|False|True|0.0"
)

# Focused follow-up around the carrier-free winner od20_e18. These groups do
# not repeat the historical projection-Mamba baseline or carrier-based ODSMF.
declare -a PURESTATE8_EXPERIMENTS=(
  "ps8_e00_dual_noscan|necessity|True|True|True|False|256|0|False|fixed|separable|False|False|False|0.1"
  "ps8_e01_consensus_scan|state|True|False|True|False|256|1|False|fixed|separable|False|False|False|0.1"
  "ps8_e02_discrepancy_scan|state|False|True|True|False|256|1|False|fixed|separable|False|False|False|0.1"
  "ps8_e03_packed_scan|topology|True|True|True|False|256|1|True|fixed|separable|False|False|False|0.1"
  "ps8_e04_channel_gate|gate|True|True|True|False|256|1|False|fixed|channel|False|False|False|0.1"
  "ps8_e05_no_gate|gate|True|True|True|False|256|1|False|fixed|none|False|False|False|0.1"
  "ps8_e06_shuffle_discrepancy|causal|True|True|True|False|256|1|False|fixed|separable|True|False|False|0.1"
  "ps8_e07_detach_discrepancy|gradient|True|True|True|False|256|1|False|fixed|separable|False|True|False|0.1"
)

if [[ "${EXPERIMENT_SUITE}" == "purestate8" ]]; then
  declare -a EXPERIMENTS=("${PURESTATE8_EXPERIMENTS[@]}")
else
  declare -a EXPERIMENTS=("${ODSMF20_EXPERIMENTS[@]}")
fi

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_FAMILY SPEC_C SPEC_D SPEC_SHARE_MAMBA \
    SPEC_SHARE_NORM SPEC_DIM SPEC_DEPTH SPEC_PACKED SPEC_BASIS SPEC_GATE \
    SPEC_SHUFFLE SPEC_DETACH SPEC_CARRIER SPEC_ALPHA <<< "$1"
}

select_specs() {
  local spec name pattern
  if [[ -z "${EXPERIMENT_FILTER}" ]]; then
    printf '%s\n' "${EXPERIMENTS[@]}"
    return
  fi
  IFS=',' read -r -a FILTERS <<< "${EXPERIMENT_FILTER}"
  for spec in "${EXPERIMENTS[@]}"; do
    name="${spec%%|*}"
    for pattern in "${FILTERS[@]}"; do
      if [[ "${name}" == *"${pattern}"* ]]; then
        printf '%s\n' "${spec}"
        break
      fi
    done
  done
}

common_opts() {
  local gpu="$1"
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
    MODEL.OSNET_FUSION.FCU_EXCHANGE_ENABLED True \
    MODEL.OSNET_FUSION.FCU_STAGES "[1,2]" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE1_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_BYPASS False \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_FORWARD_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MAMBA_MLP_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MAMBA_D_STATE 8 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_D_CONV 3 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MAMBA_SCAN_MODE "'raster'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_LEARNABLE_DIRECTION_WEIGHTS False \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_FORWARD_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_ODSMF_STATE_DIM "${SPEC_DIM}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_DEPTH "${SPEC_DEPTH}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_USE_CONSENSUS "${SPEC_C}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_USE_DISCREPANCY "${SPEC_D}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHARE_MAMBA "${SPEC_SHARE_MAMBA}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHARE_NORM "${SPEC_SHARE_NORM}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_PACKED_SCAN "${SPEC_PACKED}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_BASIS "'${SPEC_BASIS}'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_GATE "'${SPEC_GATE}'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHUFFLE_DISCREPANCY "${SPEC_SHUFFLE}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_DETACH_DISCREPANCY "${SPEC_DETACH}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_CARRIER_ENABLED "${SPEC_CARRIER}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_RES_SCALE_INIT "${SPEC_ALPHA}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_RES_SCALE_MAX 0.5 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED True \
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
    MODEL.MAMBAVISION.USE_SFM False \
    INPUT.PAM.ENABLED False \
    INPUT.DUAL_VIEW.ENABLED True \
    INPUT.DUAL_VIEW.MODE "'shared'" \
    INPUT.DUAL_VIEW.PROB 0.75 \
    INPUT.DUAL_VIEW.DIRECTION "'mamba_erased'" \
    INPUT.DUAL_VIEW.PID_BALANCED False \
    INPUT.DUAL_VIEW.CROP_PROB 0.0 \
    INPUT.DUAL_VIEW.APPEARANCE_TYPE "'none'" \
    INPUT.DUAL_VIEW.APPEARANCE_TARGET "'shared'" \
    INPUT.DUAL_VIEW.APPEARANCE_PROB 0.0 \
    INPUT.DUAL_VIEW.APPEARANCE_STRENGTH 0.0 \
    INPUT.RE_PROB 0.0 \
    INPUT.OSBBM.ENABLED False \
    SOLVER.SEED "${SEED}" \
    SOLVER.OSNET_LR_FACTOR 2.0 \
    SOLVER.OSNET_WEIGHT_DECAY 0.0005 \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS 0.0005 \
    SOLVER.OSNET_FUSION_LR_FACTOR 3.0 \
    SOLVER.FDMF_LR_FACTOR 3.0 \
    SOLVER.FCU_LR_FACTOR 3.0 \
    SOLVER.RATR_ENABLED False \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.CHECKPOINT_PERIOD 40 \
    SOLVER.EVAL_PERIOD "${MAX_EPOCHS}" \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_train() {
  local gpu="$1" spec="$2" output_dir process_log
  local -a opts=()
  read_spec "${spec}"
  output_dir="${OUTPUT_BASE}/${SPEC_NAME}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[${RUN_TAG}] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  process_log="${output_dir}/process_train.log"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[${RUN_TAG}] TRAIN gpu=${gpu} exp=${SPEC_NAME} basis=${SPEC_BASIS} gate=${SPEC_GATE} dim=${SPEC_DIM} depth=${SPEC_DEPTH}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet'" \
    OUTPUT_DIR "${output_dir}" 2>&1 | tee "${process_log}"
}

run_eval_mode() {
  local gpu="$1" weight="$2" mode="$3" eval_dir="$4" process_log
  local -a opts=()
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && \
     grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[${RUN_TAG}] SKIP eval ${SPEC_NAME}/${mode}"
    return
  fi
  mkdir -p "${eval_dir}"
  process_log="${eval_dir}/process_eval.log"
  mapfile -t opts < <(common_opts "${gpu}")
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" TEST.WEIGHT "'${weight}'" TEST.FEAT_MODE "'${mode}'" \
    TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}" \
    2>&1 | tee "${process_log}"
}

run_eval() {
  local gpu="$1" spec="$2" weight mode eval_dir
  read_spec "${spec}"
  weight="${OUTPUT_BASE}/${SPEC_NAME}/transformer_${MAX_EPOCHS}.pth"
  if [[ ! -f "${weight}" ]]; then
    echo "[${RUN_TAG}] MISSING ${weight}"
    return
  fi
  local -a modes=(
    fdmf_only fdmf weighted_mamba_fdmf_osnet
    odsmf_consensus odsmf_discrepancy odsmf_dual_state
    odsmf_dual_state_fdmf
  )
  for mode in "${modes[@]}"; do
    eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${mode}/${SPEC_NAME}"
    echo "[${RUN_TAG}] EVAL gpu=${gpu} exp=${SPEC_NAME} mode=${mode}"
    run_eval_mode "${gpu}" "${weight}" "${mode}" "${eval_dir}"
  done
}

run_workers() {
  local action="$1"
  shift
  local -a specs=("$@") pids=()
  local worker_count="${MAX_JOBS}"
  if [[ "${worker_count}" -gt "${#specs[@]}" ]]; then
    worker_count="${#specs[@]}"
  fi
  local slot
  for ((slot=0; slot<worker_count; slot++)); do
    (
      local gpu="${GPUS[$((slot % ${#GPUS[@]}))]}" idx
      for ((idx=slot; idx<${#specs[@]}; idx+=worker_count)); do
        if [[ "${action}" == "train" ]]; then
          run_train "${gpu}" "${specs[$idx]}" || \
            echo "[${RUN_TAG}] FAILED train ${specs[$idx]%%|*}" >&2
        else
          run_eval "${gpu}" "${specs[$idx]}" || \
            echo "[${RUN_TAG}] FAILED eval ${specs[$idx]%%|*}" >&2
        fi
      done
    ) &
    pids+=("$!")
  done
  local pid failures=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=1
  done
  return "${failures}"
}

summarize_specs() {
  local output_path="${OUTPUT_BASE}/summary.txt" specs
  specs="$(printf '%s\n' "$@")"
  ODSMF_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${output_path}" "${EXPERIMENT_SUITE}" <<'PY'
import os
import re
import sys

base, epoch, output_path, suite = sys.argv[1:]
specs = [x for x in os.environ.get('ODSMF_SPECS', '').splitlines() if x]
modes = [
    'fdmf_only', 'fdmf', 'weighted_mamba_fdmf_osnet',
    'odsmf_consensus', 'odsmf_discrepancy', 'odsmf_dual_state',
    'odsmf_dual_state_fdmf',
]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?\s*([0-9.]+)%')

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
    keys = ('mAP', 'R1', 'R5', 'R10')
    return result if all(key in result for key in keys) else None

rows, by_mode = [], {}
for spec in specs:
    fields = spec.split('|')
    for mode in modes:
        path = os.path.join(
            base, 'eval', f'ep{epoch}', mode, fields[0], 'test_log.txt'
        )
        result = parse(path)
        if result is not None:
            rows.append((fields, mode, result))
            by_mode[(fields[0], mode)] = result

reference_line = (
    'Internal anchor (not retrained): od20_e18_pure_dualstate, '
    'fdmf_only=79.7/89.9, weighted=84.5/91.9.'
    if suite == 'purestate8' else
    'Historical reference (not retrained): stable projection-Mamba FDMF '
    'approximately 84.5-84.8 mAP.'
)
lines = [
    reference_line,
    '',
    f"{'experiment':<34} {'family':<14} {'C/D':<5} {'share':<5} "
    f"{'norm':<5} {'dim':>4} {'dep':>3} {'pack':<5} {'basis':<18} "
    f"{'gate':<10} {'shuf':<5} {'det':<5} {'car':<5} {'a':>4} "
    f"{'mode':<29} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}",
]
for fields, mode, result in rows:
    (name, family, use_c, use_d, share, share_norm, dim, depth, packed,
     basis, gate, shuffle, detach, carrier, alpha) = fields
    states = ('C' if use_c == 'True' else '-') + ('D' if use_d == 'True' else '-')
    lines.append(
        f"{name:<34} {family:<14} {states:<5} {share:<5} {share_norm:<5} "
        f"{dim:>4} {depth:>3} {packed:<5} {basis:<18} {gate:<10} "
        f"{shuffle:<5} {detach:<5} {carrier:<5} {alpha:>4} {mode:<29} "
        f"{result['mAP']:>6.1f} {result['R1']:>6.1f} "
        f"{result['R5']:>6.1f} {result['R10']:>6.1f}"
    )

if suite == 'purestate8':
    reference = 'od20_e18_pure_dualstate'
    by_mode[(reference, 'fdmf_only')] = {'mAP': 79.7, 'R1': 89.9}
    by_mode[(reference, 'weighted_mamba_fdmf_osnet')] = {
        'mAP': 84.5, 'R1': 91.9,
    }
    comparisons = (
        ('Mamba scan contribution', reference, 'ps8_e00_dual_noscan'),
        ('Dual vs consensus only', reference, 'ps8_e01_consensus_scan'),
        ('Dual vs discrepancy only', reference, 'ps8_e02_discrepancy_scan'),
        ('Parallel vs packed scan', reference, 'ps8_e03_packed_scan'),
        ('Separable vs channel gate', reference, 'ps8_e04_channel_gate'),
        ('Separable vs no gate', reference, 'ps8_e05_no_gate'),
        ('Correct-pair discrepancy', reference, 'ps8_e06_shuffle_discrepancy'),
        ('D gradient contribution', reference, 'ps8_e07_detach_discrepancy'),
    )
else:
    comparisons = (
        ('Dual vs consensus only', 'od20_e00_dual_shared', 'od20_e01_consensus_only'),
        ('Dual vs discrepancy only', 'od20_e00_dual_shared', 'od20_e02_discrepancy_only'),
        ('Shared vs independent Mamba', 'od20_e00_dual_shared', 'od20_e03_dual_independent'),
        ('Separate vs tied norms', 'od20_e00_dual_shared', 'od20_e04_dual_tied_norm'),
        ('Width 256 vs 128', 'od20_e00_dual_shared', 'od20_e05_width128'),
        ('Width 512 vs 256', 'od20_e06_width512', 'od20_e00_dual_shared'),
        ('Depth 2 vs 1', 'od20_e07_depth2', 'od20_e00_dual_shared'),
        ('Parallel vs packed scan', 'od20_e00_dual_shared', 'od20_e08_packed_cd_scan'),
        ('Signed vs absolute D', 'od20_e00_dual_shared', 'od20_e09_absdiff'),
        ('Signed+abs vs signed D', 'od20_e10_signed_abs', 'od20_e00_dual_shared'),
        ('Learned orthogonal vs fixed', 'od20_e11_learned_orthogonal', 'od20_e00_dual_shared'),
        ('Orthogonal vs free mixing', 'od20_e11_learned_orthogonal', 'od20_e12_learned_free'),
        ('Separable gate vs no gate', 'od20_e00_dual_shared', 'od20_e13_no_gate'),
        ('Separable vs spatial gate', 'od20_e00_dual_shared', 'od20_e14_spatial_gate'),
        ('Separable vs channel gate', 'od20_e00_dual_shared', 'od20_e15_channel_gate'),
        ('Correct-pair discrepancy', 'od20_e00_dual_shared', 'od20_e16_shuffle_discrepancy'),
        ('D gradient contribution', 'od20_e00_dual_shared', 'od20_e17_detach_discrepancy'),
        ('Carrier contribution', 'od20_e00_dual_shared', 'od20_e18_pure_dualstate'),
        ('Alpha 0.1 vs zero-init', 'od20_e00_dual_shared', 'od20_e19_alpha_zero'),
    )
for mode in ('fdmf_only', 'weighted_mamba_fdmf_osnet'):
    lines.extend(['', f'Key contrasts ({mode}):'])
    lines.append(f"{'contrast':<37} {'d_mAP':>8} {'d_R1':>8}")
    for label, lhs, rhs in comparisons:
        left = by_mode.get((lhs, mode))
        right = by_mode.get((rhs, mode))
        if left is not None and right is not None:
            lines.append(
                f"{label:<37} {left['mAP'] - right['mAP']:>+8.1f} "
                f"{left['R1'] - right['R1']:>+8.1f}"
            )

text = '\n'.join(lines) + '\n'
print(text, end='')
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text)
PY
  echo "[${RUN_TAG}] Summary saved to ${output_path}"
}

mapfile -t SELECTED < <(select_specs)
if [[ "${#SELECTED[@]}" -eq 0 ]]; then
  echo "[${RUN_TAG}] No experiments selected." >&2
  exit 2
fi
echo "[${RUN_TAG}] Scheduler: jobs=${MAX_JOBS}, GPUs=${GPUS[*]}; no external GPU-process check."

case "${MODE}" in
  train)
    run_workers train "${SELECTED[@]}" || true
    ;;
  eval)
    run_workers eval "${SELECTED[@]}" || true
    ;;
  all)
    run_workers train "${SELECTED[@]}" || true
    run_workers eval "${SELECTED[@]}" || true
    summarize_specs "${SELECTED[@]}"
    ;;
  summary)
    summarize_specs "${SELECTED[@]}"
    ;;
esac
