#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Duke FDMF learning-rate audit with Mamba + outer MLP + MSEF as the primary
# structure. Stage4 and FCU are fixed at 1x unless explicitly audited.
# Four-GPU usage: bash scripts/run_duke_fdmf_lr_msef6.sh 0,1,2,3 4

MODE="${MODE:-all}"                 # train / eval / all / summary
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
MAX_JOBS="${2:-4}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_lr_msef6_s42}"
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
if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|osnet_lr|fcu_lr|fdmf_lr|msef_forward
# e00/e01/e02: FDMF LR dose; e03: OSNet LR; e04: FCU LR;
# e05: whether MSEF becomes useful after lowering all interacting-module LRs.
declare -a EXPERIMENTS=(
  "lrm6_e00_full_f1|fdmf_lr|1.0|1.0|1.0|True"
  "lrm6_e01_full_f15|fdmf_lr|1.0|1.0|1.5|True"
  "lrm6_e02_full_f3|fdmf_lr|1.0|1.0|3.0|True"
  "lrm6_e03_full_os2|osnet_lr|2.0|1.0|1.0|True"
  "lrm6_e04_full_fcu3|fcu_lr|1.0|3.0|1.0|True"
  "lrm6_e05_nomsef_f1|msef|1.0|1.0|1.0|False"
)

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_FAMILY SPEC_OSNET_LR SPEC_FCU_LR \
    SPEC_FDMF_LR SPEC_MSEF_FORWARD <<< "$1"
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
    MODEL.OSNET_FUSION.FDMF_MSEF_FORWARD_ENABLED "${SPEC_MSEF_FORWARD}" \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
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
    SOLVER.STAGE4_LR_FACTOR 1.0 \
    SOLVER.OSNET_LR_FACTOR "${SPEC_OSNET_LR}" \
    SOLVER.OSNET_WEIGHT_DECAY 0.0005 \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS 0.0005 \
    SOLVER.OSNET_FUSION_LR_FACTOR 2.0 \
    SOLVER.FCU_LR_FACTOR "${SPEC_FCU_LR}" \
    SOLVER.FDMF_LR_FACTOR "${SPEC_FDMF_LR}" \
    SOLVER.RATR_ENABLED False \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.CHECKPOINT_PERIOD 40 \
    SOLVER.EVAL_PERIOD "${MAX_EPOCHS}" \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_train() {
  local idx="$1" spec="$2" gpu output_dir
  local -a opts=()
  read_spec "${spec}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  output_dir="${OUTPUT_BASE}/${SPEC_NAME}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[FDMFLRMSEF6] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[FDMFLRMSEF6] TRAIN gpu=${gpu} exp=${SPEC_NAME} os=${SPEC_OSNET_LR} fcu=${SPEC_FCU_LR} fdmf=${SPEC_FDMF_LR} msef=${SPEC_MSEF_FORWARD}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet'" \
    OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2" gpu weight mode eval_dir
  local -a opts=()
  read_spec "${spec}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  weight="${OUTPUT_BASE}/${SPEC_NAME}/transformer_${MAX_EPOCHS}.pth"
  if [[ ! -f "${weight}" ]]; then
    echo "[FDMFLRMSEF6] MISSING ${weight}"
    return
  fi
  mapfile -t opts < <(common_opts "${gpu}")
  for mode in fdmf_only fdmf weighted_mamba_fdmf_osnet; do
    eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${mode}/${SPEC_NAME}"
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && \
       grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
      echo "[FDMFLRMSEF6] SKIP eval ${SPEC_NAME}/${mode}"
      continue
    fi
    mkdir -p "${eval_dir}"
    echo "[FDMFLRMSEF6] EVAL gpu=${gpu} exp=${SPEC_NAME} mode=${mode}"
    CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
      "${opts[@]}" TEST.WEIGHT "'${weight}'" TEST.FEAT_MODE "'${mode}'" \
      TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" \
      TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
  done
}

run_pool() {
  local action="$1"
  shift
  local -a specs=("$@") wave_pids=()
  local failures=0 idx pid
  for idx in "${!specs[@]}"; do
    if [[ "${action}" == "train" ]]; then
      run_train "${idx}" "${specs[$idx]}" &
    else
      run_eval "${idx}" "${specs[$idx]}" &
    fi
    wave_pids+=("$!")
    if [[ "${#wave_pids[@]}" -ge "${MAX_JOBS}" ]]; then
      for pid in "${wave_pids[@]}"; do wait "${pid}" || failures=1; done
      wave_pids=()
      [[ "${failures}" -eq 0 ]] || return 1
    fi
  done
  for pid in "${wave_pids[@]}"; do wait "${pid}" || failures=1; done
  return "${failures}"
}

summarize_specs() {
  local output_path="${OUTPUT_BASE}/summary.txt" specs
  specs="$(printf '%s\n' "$@")"
  FDMF_LR_MSEF6_SPECS="${specs}" python - \
    "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${output_path}" <<'PY'
import os
import re
import sys

base, epoch, output_path = sys.argv[1:]
specs = [x for x in os.environ.get('FDMF_LR_MSEF6_SPECS', '').splitlines() if x]
modes = ('fdmf_only', 'fdmf', 'weighted_mamba_fdmf_osnet')
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?\s*([0-9.]+)%')

def parse(path):
    current = {}
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            match = map_re.search(line)
            if match:
                current = {'mAP': float(match.group(1))}
                continue
            match = rank_re.search(line)
            if match and 'mAP' in current:
                current['R' + match.group(1)] = float(match.group(2))
    return current if all(k in current for k in ('mAP', 'R1', 'R5', 'R10')) else None

rows = []
by_mode = {mode: {} for mode in modes}
for spec in specs:
    fields = spec.split('|')
    for mode in modes:
        path = os.path.join(base, 'eval', f'ep{epoch}', mode, fields[0], 'test_log.txt')
        result = parse(path)
        if result is not None:
            rows.append((fields, mode, result))
            by_mode[mode][fields[0]] = result

lines = [
    f"{'experiment':<25} {'family':<9} {'os':>4} {'fcu':>4} {'fdmf_lr':>7} "
    f"{'msef':<5} {'mode':<29} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}"
]
for fields, mode, result in rows:
    name, family, os_lr, fcu_lr, fdmf_lr, msef = fields
    lines.append(
        f"{name:<25} {family:<9} {os_lr:>4} {fcu_lr:>4} {fdmf_lr:>7} "
        f"{msef:<5} {mode:<29} {result['mAP']:>6.1f} {result['R1']:>6.1f} "
        f"{result['R5']:>6.1f} {result['R10']:>6.1f}"
    )

contrasts = (
    ('FDMF 1.5x - 1x', 'lrm6_e01_full_f15', 'lrm6_e00_full_f1'),
    ('FDMF 3x - 1x', 'lrm6_e02_full_f3', 'lrm6_e00_full_f1'),
    ('OSNet 2x - 1x', 'lrm6_e03_full_os2', 'lrm6_e00_full_f1'),
    ('FCU 3x - 1x', 'lrm6_e04_full_fcu3', 'lrm6_e00_full_f1'),
    ('MSEF on - off @1x', 'lrm6_e00_full_f1', 'lrm6_e05_nomsef_f1'),
)
for mode in modes:
    table = by_mode[mode]
    available = [(label, table[a], table[b]) for label, a, b in contrasts if a in table and b in table]
    if not available:
        continue
    lines.extend(['', f'Learning-rate contrasts ({mode}):'])
    lines.append(f"{'contrast':<24} {'d_mAP':>8} {'d_R1':>8}")
    for label, lhs, rhs in available:
        lines.append(f"{label:<24} {lhs['mAP']-rhs['mAP']:>+8.1f} {lhs['R1']-rhs['R1']:>+8.1f}")

text = '\n'.join(lines)
print(text)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text + '\n')
print(f'[FDMFLRMSEF6] Summary saved to {output_path}')
PY
}

mapfile -t SELECTED_SPECS < <(select_specs)
if [[ "${#SELECTED_SPECS[@]}" -eq 0 ]]; then
  echo "No experiments matched EXPERIMENT_FILTER=${EXPERIMENT_FILTER}" >&2
  exit 2
fi

echo "[FDMFLRMSEF6] MODE=${MODE} selected=${#SELECTED_SPECS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} output=${OUTPUT_BASE}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  run_pool train "${SELECTED_SPECS[@]}"
fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  run_pool eval "${SELECTED_SPECS[@]}"
fi
if [[ "${MODE}" == "summary" || "${MODE}" == "all" ]]; then
  summarize_specs "${SELECTED_SPECS[@]}"
fi
echo "[FDMFLRMSEF6] Done."
