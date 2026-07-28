#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_local24_s42}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEARCH_SEED="${SEARCH_SEED:-42}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
# Frozen across the entire search so training variants share one retrieval descriptor.
LOCAL_INFER_WEIGHT="0.3"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|enabled|id_w|tri_w|denom|balance|order|part_id|joint_frac|tri_mode|confidence|gate_mode
declare -a EXPERIMENTS=(
  "l24_e00_nolocal_std|control|False|0.0|0.0|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e01_nolocal_d26|control|False|0.0|0.0|2.6|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e02_regonly|control|True|0.0|0.0|2.6|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e03_id10_tri10|loss|True|0.1|0.1|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e04_id10_tri00|loss|True|0.1|0.0|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e05_id00_tri10|loss|True|0.0|0.1|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e06_id10_tri05|loss|True|0.1|0.05|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e07_id20_tri10|loss|True|0.2|0.1|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e08_id20_tri05|loss|True|0.2|0.05|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e09_id30_tri10|loss|True|0.3|0.1|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e10_id30_tri05|loss|True|0.3|0.05|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e11_id10_tri20|loss|True|0.1|0.2|0.0|0.01|0.01|none|0.5|flat|none|emphasis"
  "l24_e12_noreg|regularizer|True|0.1|0.1|0.0|0.0|0.0|none|0.5|flat|none|emphasis"
  "l24_e13_balance_only|regularizer|True|0.1|0.1|0.0|0.01|0.0|none|0.5|flat|none|emphasis"
  "l24_e14_order_only|regularizer|True|0.1|0.1|0.0|0.0|0.01|none|0.5|flat|none|emphasis"
  "l24_e15_reg_strong|regularizer|True|0.1|0.1|0.0|0.05|0.05|none|0.5|flat|none|emphasis"
  "l24_e16_partid_flatri|part|True|0.1|0.1|0.0|0.01|0.01|replace|0.5|flat|none|emphasis"
  "l24_e17_joint_partid_flatri|part|True|0.1|0.1|0.0|0.01|0.01|joint|0.5|flat|none|emphasis"
  "l24_e18_flatid_partavgtri|part|True|0.1|0.1|0.0|0.01|0.01|none|0.5|part_avg|none|emphasis"
  "l24_e19_partid_partavgtri|part|True|0.1|0.1|0.0|0.01|0.01|replace|0.5|part_avg|none|emphasis"
  "l24_e20_conf_partavgtri|confidence|True|0.1|0.1|0.0|0.01|0.01|none|0.5|part_avg|triplet|emphasis"
  "l24_e21_conf_descriptor|confidence|True|0.1|0.1|0.0|0.01|0.01|none|0.5|part_avg|descriptor|emphasis"
  "l24_e22_gate_2sigmoid|gate|True|0.1|0.1|0.0|0.01|0.01|none|0.5|flat|none|sigmoid2"
  "l24_e23_gate_floor01|gate|True|0.1|0.1|0.0|0.01|0.01|none|0.5|flat|none|floor01"
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
  local gpu="$1" enabled="$2" id_w="$3" tri_w="$4" denom="$5"
  local balance="$6" order="$7" part_id="$8" joint_frac="$9"
  local tri_mode="${10}" confidence="${11}" gate_mode="${12}"
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
    MODEL.OSNET_FUSION.STAGE3_LOCAL_PART_ID_MODE "'${part_id}'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_PART_ID_JOINT_FRACTION "${joint_frac}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TRIPLET_MODE "'${tri_mode}'" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_CONFIDENCE_MODE "'${confidence}'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${LOCAL_INFER_WEIGHT}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE 1.5 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE 1.0 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_BALANCE_WEIGHT "${balance}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_WEIGHT "${order}" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE True \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE_MODE "'${gate_mode}'" \
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
  local idx="$1" spec="$2"
  local name family enabled id_w tri_w denom balance order part_id joint_frac tri_mode confidence gate_mode
  IFS='|' read -r name family enabled id_w tri_w denom balance order part_id joint_frac tri_mode confidence gate_mode <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" output_dir="${OUTPUT_BASE}/${name}"
  local feature='weighted_mamba_fdmf_osnet_stage3local' opts=()
  [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[Local24] SKIP train ${name}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${id_w}" "${tri_w}" "${denom}" "${balance}" "${order}" "${part_id}" "${joint_frac}" "${tri_mode}" "${confidence}" "${gate_mode}")
  echo "[Local24] TRAIN gpu=${gpu} exp=${name} family=${family} id=${id_w} tri=${tri_w} part=${part_id}/${tri_mode} conf=${confidence} gate=${gate_mode}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" \
    TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2"
  local name family enabled id_w tri_w denom balance order part_id joint_frac tri_mode confidence gate_mode
  IFS='|' read -r name family enabled id_w tri_w denom balance order part_id joint_frac tri_mode confidence gate_mode <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${OUTPUT_BASE}/${name}/transformer_${MAX_EPOCHS}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${MAX_EPOCHS}/${name}"
  local feature='weighted_mamba_fdmf_osnet_stage3local' opts=()
  [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  [[ -f "${weight}" ]] || { echo "[Local24] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[Local24] SKIP eval ${name}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${id_w}" "${tri_w}" "${denom}" "${balance}" "${order}" "${part_id}" "${joint_frac}" "${tri_mode}" "${confidence}" "${gate_mode}")
  echo "[Local24] EVAL gpu=${gpu} exp=${name} local_iw=${LOCAL_INFER_WEIGHT}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEARCH_SEED}" TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'${feature}'" TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" \
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
  LOCAL24_SPECS="${specs}" python - "${OUTPUT_BASE}" "${MAX_EPOCHS}" <<'PY'
import os
import re
import sys

base, epoch = sys.argv[1], int(sys.argv[2])
specs = [line for line in os.environ.get('LOCAL24_SPECS', '').splitlines() if line]
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

print(
    f"{'experiment':<34} {'family':<11} {'id':>4} {'tri':>4} "
    f"{'part_id':<7} {'tri_mode':<8} {'conf':<10} {'gate':<9} "
    f"{'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}"
)
final = []
for spec in specs:
    fields = spec.split('|')
    name, family, enabled, id_w, tri_w, denom, balance, order, part_id, joint_frac, tri_mode, confidence, gate_mode = fields
    path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
    result = parse(path) if os.path.exists(path) else None
    values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
    print(
        f"{name:<34} {family:<11} {id_w:>4} {tri_w:>4} "
        f"{part_id:<7} {tri_mode:<8} {confidence:<10} {gate_mode:<9} "
        f"{values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}"
    )
    if result is not None:
        final.append((result['mAP'], result['R1'], name, family))
if final:
    best_map = max(final, key=lambda row: (row[0], row[1]))
    best_r1 = max(final, key=lambda row: (row[1], row[0]))
    print(f"\nbest_mAP: {best_map[2]} family={best_map[3]} mAP={best_map[0]:.1f} R1={best_map[1]:.1f}")
    print(f"best_R1 : {best_r1[2]} family={best_r1[3]} mAP={best_r1[0]:.1f} R1={best_r1[1]:.1f}")
PY
}

echo "[Local24] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEARCH_SEED} local_iw=${LOCAL_INFER_WEIGHT}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[Local24] Summary saved to ${OUTPUT_BASE}/summary.txt"
