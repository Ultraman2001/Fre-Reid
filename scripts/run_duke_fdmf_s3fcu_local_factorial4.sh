#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Matched 2x2 causal test for the possible redundancy between:
#   1) Stage3 Mamba -> OSNet FCU exchange, and
#   2) the Stage3 semantic-detail local auxiliary supervision.
# Stage2 OSNet -> Mamba exchange stays enabled in all four groups.
# All groups use the same global/FDMF descriptor at inference; the local
# descriptor is deliberately excluded so that only its training effect is
# measured.

MODE="${MODE:-all}"                 # train / eval / all / summary
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_s3fcu_local_factorial4_s42}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SEARCH_SEED="${SEARCH_SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
EVAL_DECOMPOSE="${EVAL_DECOMPOSE:-1}"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|stage3_fcu|local
declare -a EXPERIMENTS=(
  "s3l4_e00_s3off_localoff|False|False"
  "s3l4_e01_s3on_localoff|True|False"
  "s3l4_e02_s3off_localon|False|True"
  "s3l4_e03_s3on_localon|True|True"
)

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_STAGE3_FCU SPEC_LOCAL <<< "$1"
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
  local fcu_stages="[2]"
  local -a pretrain_opts=()
  [[ "${SPEC_STAGE3_FCU}" == "True" ]] && fcu_stages="[2,3]"
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
    MODEL.OSNET_FUSION.FCU_STAGES "${fcu_stages}" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'mamba_to_osnet'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_BYPASS False \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MAMBA_SCAN_MODE "'raster'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_LEARNABLE_DIRECTION_WEIGHTS False \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.COMPLEMENTARITY.MODE "'none'" \
    MODEL.OSNET_FUSION.PEER_COMPLEMENT.ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${SPEC_LOCAL}" \
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
    MODEL.OSNET_FUSION.ROLE_SPECIALIZATION.ENABLED False \
    INPUT.PAM.ENABLED False \
    INPUT.DUAL_VIEW.ENABLED True \
    INPUT.DUAL_VIEW.MODE "'shared'" \
    INPUT.DUAL_VIEW.PROB 0.75 \
    INPUT.DUAL_VIEW.DIRECTION "'mamba_erased'" \
    INPUT.DUAL_VIEW.PID_BALANCED False \
    INPUT.DUAL_VIEW.CROP_PROB 0.0 \
    INPUT.RE_PROB 0.0 \
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
  local -a opts=()
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[S3Local4] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[S3Local4] TRAIN gpu=${gpu} exp=${SPEC_NAME} stage3_fcu=${SPEC_STAGE3_FCU} local=${SPEC_LOCAL}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
    TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet'" \
    OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2"
  read_spec "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${OUTPUT_BASE}/${SPEC_NAME}/transformer_${MAX_EPOCHS}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${SPEC_NAME}"
  local -a opts=()
  local -a labels=(main)
  local -a features=(weighted_mamba_fdmf_osnet)
  if [[ "${EVAL_DECOMPOSE}" == "1" ]]; then
    labels+=(mamba osnet fdmf)
    features+=(backbone osnet fdmf_only)
    if [[ "${SPEC_LOCAL}" == "True" ]]; then
      labels+=(local)
      features+=(stage3_stripe_local)
    fi
  fi
  if [[ ! -f "${weight}" ]]; then
    echo "[S3Local4] MISSING ${weight}"
    return
  fi
  mapfile -t opts < <(common_opts "${gpu}")
  local i label feature mode_dir
  for i in "${!labels[@]}"; do
    label="${labels[$i]}"
    feature="${features[$i]}"
    mode_dir="${eval_dir}"
    [[ "${label}" == "main" ]] || mode_dir="${eval_dir}/${label}"
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${mode_dir}/test_log.txt" ]] && grep -q "Rank-10" "${mode_dir}/test_log.txt"; then
      echo "[S3Local4] SKIP eval ${SPEC_NAME}/${label}"
      continue
    fi
    mkdir -p "${mode_dir}"
    echo "[S3Local4] EVAL gpu=${gpu} exp=${SPEC_NAME} mode=${label}/${feature}"
    CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
      "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
      TEST.WEIGHT "'${weight}'" \
      TEST.FEAT_MODE "'${feature}'" \
      TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" \
      TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${mode_dir}"
  done
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
  local output_path="${OUTPUT_BASE}/summary.txt"
  local specs
  specs="$(printf '%s\n' "$@")"
  S3LOCAL4_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${output_path}" <<'PY'
import os
import re
import sys

base, epoch, output_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
specs = [x for x in os.environ.get('S3LOCAL4_SPECS', '').splitlines() if x]
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
    keys = ('mAP', 'R1', 'R5', 'R10')
    return result if all(key in result for key in keys) else None

rows = []
by_state = {}
for spec in specs:
    name, stage3_fcu, local = spec.split('|')
    result = parse(os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt'))
    if result is not None:
        rows.append((name, stage3_fcu, local, result))
        by_state[(stage3_fcu, local)] = result

lines = [
    f"{'experiment':<30} {'stage3_fcu':<10} {'local_train':<11} "
    f"{'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}"
]
for name, stage3_fcu, local, result in rows:
    lines.append(
        f"{name:<30} {stage3_fcu:<10} {local:<11} "
        f"{result['mAP']:>6.1f} {result['R1']:>6.1f} "
        f"{result['R5']:>6.1f} {result['R10']:>6.1f}"
    )

states = [('False', 'False'), ('True', 'False'), ('False', 'True'), ('True', 'True')]
if all(state in by_state for state in states):
    a = by_state[('False', 'False')]
    b = by_state[('True', 'False')]
    c = by_state[('False', 'True')]
    d = by_state[('True', 'True')]
    lines.extend(['', 'Factorial decomposition (Stage2 FCU remains on in every group):'])
    lines.append(f"{'effect':<34} {'d_mAP':>8} {'d_R1':>8}")
    effects = [
        ('Stage3 FCU | local off', b, a),
        ('Stage3 FCU | local on', d, c),
        ('Local supervision | Stage3 off', c, a),
        ('Local supervision | Stage3 on', d, b),
    ]
    for label, lhs, rhs in effects:
        lines.append(
            f"{label:<34} {lhs['mAP'] - rhs['mAP']:>+8.1f} "
            f"{lhs['R1'] - rhs['R1']:>+8.1f}"
        )
    interaction_map = d['mAP'] - b['mAP'] - c['mAP'] + a['mAP']
    interaction_r1 = d['R1'] - b['R1'] - c['R1'] + a['R1']
    lines.append(
        f"{'interaction S = both-FCU-local+base':<34} "
        f"{interaction_map:>+8.1f} {interaction_r1:>+8.1f}"
    )
    lines.extend([
        '',
        'Interpretation: S>0 synergy; S~=0 additive/independent; S<0 redundancy or competition.',
        'The main descriptor excludes the local descriptor in all four groups.',
    ])

text = '\n'.join(lines)
print(text)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text + '\n')
PY
  echo "[S3Local4] Summary saved to ${output_path}"
}

mapfile -t SPECS < <(select_specs)
if [[ "${#SPECS[@]}" -eq 0 ]]; then
  echo "No experiments matched EXPERIMENT_FILTER=${EXPERIMENT_FILTER}" >&2
  exit 2
fi

echo "[S3Local4] MODE=${MODE} experiments=${#SPECS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEARCH_SEED}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  run_pool train "${SPECS[@]}"
fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  run_pool eval "${SPECS[@]}"
fi
if [[ "${MODE}" == "summary" || "${MODE}" == "all" ]]; then
  summarize_specs "${SPECS[@]}"
fi
