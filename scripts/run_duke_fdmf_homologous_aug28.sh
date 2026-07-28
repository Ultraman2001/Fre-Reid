#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Twenty-eight single-seed experiments for spatially aligned, homologous
# augmentation of the heterogeneous MambaVision/OSNet inputs. Geometry is
# sampled once before the two views are copied. Only appearance can differ.
# The established full Stage2+Stage3 FCU and semantic-detail local auxiliary
# branch stay fixed; the local descriptor is excluded from inference.

MODE="${MODE:-all}"                 # train / eval / all / summary
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_homologous_aug28_s42}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SEARCH_SEED="${SEARCH_SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|shared_erase_prob|appearance_type|target|appearance_prob|strength
# A: controls/shared appearance; B-D: target and probability; E: strength;
# F: appearance-only controls for interaction with shared erasing.
declare -a EXPERIMENTS=(
  "ha28_a00_current_p75|control|0.75|none|shared|0.0|0.0"
  "ha28_a01_shared_color|shared|0.75|color|shared|0.50|0.20"
  "ha28_a02_shared_gray|shared|0.75|grayscale|shared|0.50|0.50"
  "ha28_a03_shared_blur|shared|0.75|blur|shared|0.50|1.00"

  "ha28_b00_color_m_p25|color_prob|0.75|color|mamba|0.25|0.20"
  "ha28_b01_color_m_p50|color_prob|0.75|color|mamba|0.50|0.20"
  "ha28_b02_color_m_p75|color_prob|0.75|color|mamba|0.75|0.20"
  "ha28_b03_color_o_p50|color_target|0.75|color|osnet|0.50|0.20"
  "ha28_b04_color_rand_p50|color_target|0.75|color|random_one|0.50|0.20"

  "ha28_c00_gray_m_p25|gray_prob|0.75|grayscale|mamba|0.25|0.50"
  "ha28_c01_gray_m_p50|gray_prob|0.75|grayscale|mamba|0.50|0.50"
  "ha28_c02_gray_m_p75|gray_prob|0.75|grayscale|mamba|0.75|0.50"
  "ha28_c03_gray_o_p50|gray_target|0.75|grayscale|osnet|0.50|0.50"
  "ha28_c04_gray_rand_p50|gray_target|0.75|grayscale|random_one|0.50|0.50"

  "ha28_d00_blur_m_p25|blur_prob|0.75|blur|mamba|0.25|1.00"
  "ha28_d01_blur_m_p50|blur_prob|0.75|blur|mamba|0.50|1.00"
  "ha28_d02_blur_m_p75|blur_prob|0.75|blur|mamba|0.75|1.00"
  "ha28_d03_blur_o_p50|blur_target|0.75|blur|osnet|0.50|1.00"
  "ha28_d04_blur_rand_p50|blur_target|0.75|blur|random_one|0.50|1.00"

  "ha28_e00_color_m_s10|strength|0.75|color|mamba|0.50|0.10"
  "ha28_e01_color_m_s40|strength|0.75|color|mamba|0.50|0.40"
  "ha28_e02_gray_m_s25|strength|0.75|grayscale|mamba|0.50|0.25"
  "ha28_e03_gray_m_s75|strength|0.75|grayscale|mamba|0.50|0.75"
  "ha28_e04_blur_m_s05|strength|0.75|blur|mamba|0.50|0.50"
  "ha28_e05_blur_m_s15|strength|0.75|blur|mamba|0.50|1.50"

  "ha28_f00_color_m_re00|erase_inter|0.00|color|mamba|0.50|0.20"
  "ha28_f01_gray_m_re00|erase_inter|0.00|grayscale|mamba|0.50|0.50"
  "ha28_f02_blur_m_re00|erase_inter|0.00|blur|mamba|0.50|1.00"
)

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_FAMILY SPEC_ERASE_PROB SPEC_APP_TYPE \
    SPEC_APP_TARGET SPEC_APP_PROB SPEC_APP_STRENGTH <<< "$1"
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
    MODEL.OSNET_FUSION.FCU_STAGES "[2,3]" \
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
    MODEL.OSNET_FUSION.ROLE_SPECIALIZATION.ENABLED False \
    INPUT.PAM.ENABLED False \
    INPUT.DUAL_VIEW.ENABLED True \
    INPUT.DUAL_VIEW.MODE "'shared'" \
    INPUT.DUAL_VIEW.PROB "${SPEC_ERASE_PROB}" \
    INPUT.DUAL_VIEW.DIRECTION "'mamba_erased'" \
    INPUT.DUAL_VIEW.PID_BALANCED False \
    INPUT.DUAL_VIEW.CROP_PROB 0.0 \
    INPUT.DUAL_VIEW.APPEARANCE_TYPE "'${SPEC_APP_TYPE}'" \
    INPUT.DUAL_VIEW.APPEARANCE_TARGET "'${SPEC_APP_TARGET}'" \
    INPUT.DUAL_VIEW.APPEARANCE_PROB "${SPEC_APP_PROB}" \
    INPUT.DUAL_VIEW.APPEARANCE_STRENGTH "${SPEC_APP_STRENGTH}" \
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
    echo "[HomologousAug28] SKIP train ${SPEC_NAME}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[HomologousAug28] TRAIN gpu=${gpu} exp=${SPEC_NAME} family=${SPEC_FAMILY} erase=${SPEC_ERASE_PROB} appearance=${SPEC_APP_TYPE}/${SPEC_APP_TARGET}/p${SPEC_APP_PROB}/s${SPEC_APP_STRENGTH}"
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
  if [[ ! -f "${weight}" ]]; then
    echo "[HomologousAug28] MISSING ${weight}"
    return
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[HomologousAug28] SKIP eval ${SPEC_NAME}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}")
  echo "[HomologousAug28] EVAL gpu=${gpu} exp=${SPEC_NAME} feature=weighted_mamba_fdmf_osnet"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
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
  HOMOLOGOUS_AUG28_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" "${output_path}" <<'PY'
import os
import re
import sys

base, epoch, output_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
specs = [line for line in os.environ.get('HOMOLOGOUS_AUG28_SPECS', '').splitlines() if line]
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
for spec in specs:
    fields = spec.split('|')
    result = parse(os.path.join(base, 'eval', f'ep{epoch}', fields[0], 'test_log.txt'))
    if result is not None:
        rows.append((fields, result))

header = (
    f"{'experiment':<28} {'family':<13} {'erase':>6} {'type':<10} "
    f"{'target':<10} {'p':>5} {'strength':>8} {'mAP':>6} {'R1':>6} "
    f"{'R5':>6} {'R10':>6}"
)
lines = [header]
for fields, result in rows:
    name, family, erase, aug_type, target, prob, strength = fields
    lines.append(
        f"{name:<28} {family:<13} {erase:>6} {aug_type:<10} "
        f"{target:<10} {prob:>5} {strength:>8} "
        f"{result['mAP']:>6.1f} {result['R1']:>6.1f} "
        f"{result['R5']:>6.1f} {result['R10']:>6.1f}"
    )
if rows:
    best_map = max(rows, key=lambda item: (item[1]['mAP'], item[1]['R1']))
    best_r1 = max(rows, key=lambda item: (item[1]['R1'], item[1]['mAP']))
    lines.append('')
    lines.append(
        f"best_mAP: {best_map[0][0]} mAP={best_map[1]['mAP']:.1f} "
        f"R1={best_map[1]['R1']:.1f}"
    )
    lines.append(
        f"best_R1 : {best_r1[0][0]} mAP={best_r1[1]['mAP']:.1f} "
        f"R1={best_r1[1]['R1']:.1f}"
    )
text = '\n'.join(lines)
print(text)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text + '\n')
print(f'[HomologousAug28] Summary saved to {output_path}')
PY
}

mapfile -t SELECTED_SPECS < <(select_specs)
if [[ "${#SELECTED_SPECS[@]}" -eq 0 ]]; then
  echo "No experiments matched EXPERIMENT_FILTER=${EXPERIMENT_FILTER}" >&2
  exit 2
fi

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  run_pool train "${SELECTED_SPECS[@]}"
fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  run_pool eval "${SELECTED_SPECS[@]}"
fi
if [[ "${MODE}" == "summary" || "${MODE}" == "all" ]]; then
  summarize_specs "${SELECTED_SPECS[@]}"
fi
