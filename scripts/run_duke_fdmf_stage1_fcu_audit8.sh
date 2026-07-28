#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Eight Duke experiments that close four open questions:
#   1) Stage1 x Stage2 FCU hierarchy with local supervision (e00-e03, seed42);
#   2) Stage1+2 second-seed confirmation (e04, seed3407);
#   3) missing Stage2-only/local-off seed3407 control (e05);
#   4) two exact A00 reproductions for the 84.9/92.3 audit (e06-e07).
# The two audit replicas are trained serially after the six search/control runs.

MODE="${MODE:-all}"                 # train / eval / all / summary
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_stage1_fcu_audit8}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
AUDIT_SERIAL="${AUDIT_SERIAL:-1}"
REFERENCE_STAGE2_S3407="${REFERENCE_STAGE2_S3407:-./logs/Duke/fdmf_stage3_local_final2_s3407}"
HISTORICAL_A00_MAP="${HISTORICAL_A00_MAP:-84.9}"
HISTORICAL_A00_R1="${HISTORICAL_A00_R1:-92.3}"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|fcu_stages|exchange|local|seed|audit
declare -a EXPERIMENTS=(
  "s1a8_e00_noexchange_localon_s42|hierarchy|[2]|False|True|42|False"
  "s1a8_e01_s1_localon_s42|hierarchy|[1]|True|True|42|False"
  "s1a8_e02_s2_localon_s42|hierarchy|[2]|True|True|42|False"
  "s1a8_e03_s12_localon_s42|hierarchy|[1,2]|True|True|42|False"
  "s1a8_e04_s12_localon_s3407|stage1_repro|[1,2]|True|True|3407|False"
  "s1a8_e05_s2_localoff_s3407|local_control|[2]|True|False|3407|False"
  "s1a8_e06_a00_reproA_s42|a00_audit|[2,3]|True|True|42|True"
  "s1a8_e07_a00_reproB_s42|a00_audit|[2,3]|True|True|42|True"
)

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_FAMILY SPEC_STAGES SPEC_EXCHANGE \
    SPEC_LOCAL SPEC_SEED SPEC_AUDIT <<< "$1"
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
    MODEL.OSNET_FUSION.FCU_EXCHANGE_ENABLED "${SPEC_EXCHANGE}" \
    MODEL.OSNET_FUSION.FCU_STAGES "${SPEC_STAGES}" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE1_DIRECTION "'osnet_to_mamba'" \
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
    INPUT.DUAL_VIEW.APPEARANCE_TYPE "'none'" \
    INPUT.DUAL_VIEW.APPEARANCE_TARGET "'shared'" \
    INPUT.DUAL_VIEW.APPEARANCE_PROB 0.0 \
    INPUT.DUAL_VIEW.APPEARANCE_STRENGTH 0.0 \
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
    echo "[Stage1Audit8] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[Stage1Audit8] TRAIN gpu=${gpu} exp=${SPEC_NAME} family=${SPEC_FAMILY} stages=${SPEC_STAGES} exchange=${SPEC_EXCHANGE} local=${SPEC_LOCAL} seed=${SPEC_SEED} audit=${SPEC_AUDIT}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SPEC_SEED}" \
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
  if [[ ! -f "${weight}" ]]; then
    echo "[Stage1Audit8] MISSING ${weight}"
    return
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[Stage1Audit8] SKIP eval ${SPEC_NAME}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[Stage1Audit8] EVAL gpu=${gpu} exp=${SPEC_NAME} seed=${SPEC_SEED}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SPEC_SEED}" \
    TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet'" \
    TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local action="$1"
  shift
  local -a specs=("$@")
  local running=0 failures=0 idx
  [[ "${#specs[@]}" -gt 0 ]] || return 0
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
  STAGE1_AUDIT8_SPECS="${specs}" python - \
    "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${output_path}" \
    "${REFERENCE_STAGE2_S3407}" "${HISTORICAL_A00_MAP}" \
    "${HISTORICAL_A00_R1}" <<'PY'
import os
import re
import sys

base, epoch, output_path, reference_s3407, historical_map, historical_r1 = sys.argv[1:]
epoch = int(epoch)
historical_map = float(historical_map)
historical_r1 = float(historical_r1)
specs = [line for line in os.environ.get('STAGE1_AUDIT8_SPECS', '').splitlines() if line]
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
by_name = {}
for spec in specs:
    fields = spec.split('|')
    result = parse(os.path.join(base, 'eval', f'ep{epoch}', fields[0], 'test_log.txt'))
    if result is not None:
        rows.append((fields, result))
        by_name[fields[0]] = result

reference_stage2_s3407 = parse(os.path.join(
    reference_s3407,
    'eval',
    f'ep{epoch}',
    's3l4_e02_s3off_localon',
    'test_log.txt',
))

lines = [
    f"{'experiment':<34} {'family':<14} {'stages':<7} {'exchange':<8} "
    f"{'local':<5} {'seed':>5} {'audit':<5} {'mAP':>6} {'R1':>6} "
    f"{'R5':>6} {'R10':>6}"
]
for fields, result in rows:
    name, family, stages, exchange, local, seed, audit = fields
    lines.append(
        f"{name:<34} {family:<14} {stages:<7} {exchange:<8} "
        f"{local:<5} {seed:>5} {audit:<5} {result['mAP']:>6.1f} "
        f"{result['R1']:>6.1f} {result['R5']:>6.1f} {result['R10']:>6.1f}"
    )

hierarchy_names = (
    's1a8_e00_noexchange_localon_s42',
    's1a8_e01_s1_localon_s42',
    's1a8_e02_s2_localon_s42',
    's1a8_e03_s12_localon_s42',
)
if all(name in by_name for name in hierarchy_names):
    no_fcu, s1, s2, s12 = (by_name[name] for name in hierarchy_names)
    lines.extend(['', 'Stage1 x Stage2 hierarchy decomposition (seed42, local on):'])
    lines.append(f"{'effect':<32} {'d_mAP':>8} {'d_R1':>8}")
    effects = (
        ('Stage1 | Stage2 off', s1, no_fcu),
        ('Stage1 | Stage2 on', s12, s2),
        ('Stage2 | Stage1 off', s2, no_fcu),
        ('Stage2 | Stage1 on', s12, s1),
    )
    for label, lhs, rhs in effects:
        lines.append(
            f"{label:<32} {lhs['mAP'] - rhs['mAP']:>+8.1f} "
            f"{lhs['R1'] - rhs['R1']:>+8.1f}"
        )
    interaction_map = s12['mAP'] - s1['mAP'] - s2['mAP'] + no_fcu['mAP']
    interaction_r1 = s12['R1'] - s1['R1'] - s2['R1'] + no_fcu['R1']
    lines.append(
        f"{'interaction S1xS2':<32} {interaction_map:>+8.1f} "
        f"{interaction_r1:>+8.1f}"
    )

s12_s3407 = by_name.get('s1a8_e04_s12_localon_s3407')
if reference_stage2_s3407 is not None and s12_s3407 is not None:
    lines.extend(['', 'Stage1 marginal effect with Stage2 on (seed3407):'])
    lines.append(
        f"d_mAP={s12_s3407['mAP'] - reference_stage2_s3407['mAP']:+.1f} "
        f"d_R1={s12_s3407['R1'] - reference_stage2_s3407['R1']:+.1f}"
    )

local_off_s3407 = by_name.get('s1a8_e05_s2_localoff_s3407')
if reference_stage2_s3407 is not None and local_off_s3407 is not None:
    lines.extend(['', 'Local supervision effect with Stage2-only (seed3407, on - off):'])
    lines.append(
        f"d_mAP={reference_stage2_s3407['mAP'] - local_off_s3407['mAP']:+.1f} "
        f"d_R1={reference_stage2_s3407['R1'] - local_off_s3407['R1']:+.1f}"
    )

audit_results = [
    by_name[name]
    for name in (
        's1a8_e06_a00_reproA_s42',
        's1a8_e07_a00_reproB_s42',
    )
    if name in by_name
]
if audit_results:
    mean_map = sum(x['mAP'] for x in audit_results) / len(audit_results)
    mean_r1 = sum(x['R1'] for x in audit_results) / len(audit_results)
    spread_map = max(x['mAP'] for x in audit_results) - min(x['mAP'] for x in audit_results)
    spread_r1 = max(x['R1'] for x in audit_results) - min(x['R1'] for x in audit_results)
    lines.extend(['', 'A00 84.9/92.3 audit:'])
    lines.append(
        f"replicas={len(audit_results)} mean_mAP={mean_map:.2f} "
        f"mean_R1={mean_r1:.2f} spread_mAP={spread_map:.2f} "
        f"spread_R1={spread_r1:.2f}"
    )
    lines.append(
        f"vs_historical d_mAP={mean_map - historical_map:+.2f} "
        f"d_R1={mean_r1 - historical_r1:+.2f}"
    )

text = '\n'.join(lines)
print(text)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text + '\n')
print(f'[Stage1Audit8] Summary saved to {output_path}')
PY
}

mapfile -t SELECTED_SPECS < <(select_specs)
if [[ "${#SELECTED_SPECS[@]}" -eq 0 ]]; then
  echo "No experiments matched EXPERIMENT_FILTER=${EXPERIMENT_FILTER}" >&2
  exit 2
fi

SEARCH_SPECS=()
AUDIT_SPECS=()
for spec in "${SELECTED_SPECS[@]}"; do
  read_spec "${spec}"
  if [[ "${SPEC_AUDIT}" == "True" ]]; then
    AUDIT_SPECS+=("${spec}")
  else
    SEARCH_SPECS+=("${spec}")
  fi
done

echo "[Stage1Audit8] MODE=${MODE} selected=${#SELECTED_SPECS[@]} search=${#SEARCH_SPECS[@]} audit=${#AUDIT_SPECS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} audit_serial=${AUDIT_SERIAL}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  run_pool train "${SEARCH_SPECS[@]}"
  if [[ "${AUDIT_SERIAL}" == "1" ]]; then
    for spec in "${AUDIT_SPECS[@]}"; do
      run_train 0 "${spec}"
    done
  else
    run_pool train "${AUDIT_SPECS[@]}"
  fi
fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  run_pool eval "${SELECTED_SPECS[@]}"
fi
if [[ "${MODE}" == "summary" || "${MODE}" == "all" ]]; then
  summarize_specs "${SELECTED_SPECS[@]}"
fi
