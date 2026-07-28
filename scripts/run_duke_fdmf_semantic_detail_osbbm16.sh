#!/usr/bin/env bash
# Sixteen seed-42 OSBBM studies on the confirmed Semantic Detail backbone.
# Only epoch 160 is checkpointed and evaluated.
#
# Usage:
#   bash scripts/run_duke_fdmf_semantic_detail_osbbm16.sh 0,1 4
#   MODE=train bash scripts/run_duke_fdmf_semantic_detail_osbbm16.sh 0,1 4
#   MODE=eval  bash scripts/run_duke_fdmf_semantic_detail_osbbm16.sh 0,1 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_semantic_detail_osbbm16}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
SEED="${SEED:-42}"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|prob|blocks|mix|gray|gray_scope|sample|donor|block|schedule|start|end|period|on
declare -a EXPERIMENTS=(
  "os16_legacy_always|control|0.50|8|2|0.50|all|random|random|random|always|1|160|20|10"
  "os16_legacy_g0|control|0.50|8|2|0.00|all|random|random|random|always|1|160|20|10"

  "os16_range21_120|schedule|0.50|8|2|0.50|all|random|random|random|range|21|120|20|10"
  "os16_range41_120|schedule|0.50|8|2|0.50|all|random|random|random|range|41|120|20|10"
  "os16_cycle21_120|schedule|0.50|8|2|0.50|all|random|random|random|cycle|21|120|20|10"
  "os16_cycle21_140|schedule|0.50|8|2|0.50|all|random|random|random|cycle|21|140|20|10"

  "os16_cycle_g0|gray|0.50|8|2|0.00|all|random|random|random|cycle|21|120|20|10"
  "os16_cycle_g05_mixed|gray|0.50|8|2|0.50|mixed|random|random|random|cycle|21|120|20|10"

  "os16_cycle_p025|strength|0.25|8|2|0.00|all|random|random|random|cycle|21|120|20|10"
  "os16_cycle_p075|strength|0.75|8|2|0.00|all|random|random|random|cycle|21|120|20|10"
  "os16_cycle_m1|strength|0.50|8|1|0.00|all|random|random|random|cycle|21|120|20|10"
  "os16_cycle_m3|strength|0.50|8|3|0.00|all|random|random|random|cycle|21|120|20|10"

  "os16_pkhalf|sampling|0.50|8|2|0.00|all|pk_half|random|random|cycle|21|120|20|10"
  "os16_derange|sampling|0.50|8|2|0.00|all|random|derangement|random|cycle|21|120|20|10"
  "os16_balanced|sampling|0.50|8|2|0.00|all|random|random|part_balanced|cycle|21|120|20|10"
  "os16_structured|sampling|0.50|8|2|0.00|all|pk_half|derangement|part_balanced|cycle|21|120|20|10"
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
  local gpu="$1" prob="$2" blocks="$3" mix="$4" gray="$5" gray_scope="$6"
  local sample="$7" donor="$8" block="$9" schedule="${10}" start="${11}" end="${12}"
  local period="${13}" on="${14}"
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
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.COMPLEMENTARITY.MODE "'none'" \
    MODEL.OSNET_FUSION.PEER_COMPLEMENT.ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED True \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TYPE "'semantic_detail'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE 1.5 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE 1.0 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_BALANCE_WEIGHT 0.01 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_WEIGHT 0.01 \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_MASK_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE True \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_SOURCE "'conv4'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_RESIDUAL_INJECTION False \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED True \
    INPUT.OSBBM.PROB "${prob}" \
    INPUT.OSBBM.NUM_BLOCKS "${blocks}" \
    INPUT.OSBBM.NUM_MIX_BLOCKS "${mix}" \
    INPUT.OSBBM.GRAY_PROB "${gray}" \
    INPUT.OSBBM.GRAY_SCOPE "'${gray_scope}'" \
    INPUT.OSBBM.APPLY_TO "'base'" \
    INPUT.OSBBM.MIXED_LABEL True \
    INPUT.OSBBM.SAMPLE_MODE "'${sample}'" \
    INPUT.OSBBM.DONOR_MODE "'${donor}'" \
    INPUT.OSBBM.BLOCK_MODE "'${block}'" \
    INPUT.OSBBM.SCHEDULE "'${schedule}'" \
    INPUT.OSBBM.START_EPOCH "${start}" \
    INPUT.OSBBM.END_EPOCH "${end}" \
    INPUT.OSBBM.PERIOD_EPOCHS "${period}" \
    INPUT.OSBBM.ON_EPOCHS "${on}" \
    MODEL.MAMBAVISION.USE_SFM False \
    SOLVER.OSNET_LR_FACTOR 2.0 \
    SOLVER.OSNET_WEIGHT_DECAY 0.0005 \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS 0.0005 \
    SOLVER.OSNET_FUSION_LR_FACTOR 3.0 \
    SOLVER.RATR_ENABLED False \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.CHECKPOINT_PERIOD "${MAX_EPOCHS}" \
    SOLVER.EVAL_PERIOD "${MAX_EPOCHS}" \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_train() {
  local idx="$1" spec="$2"
  local name family prob blocks mix gray gray_scope sample donor block schedule start end period on
  IFS='|' read -r name family prob blocks mix gray gray_scope sample donor block schedule start end period on <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" output_dir="${OUTPUT_BASE}/${name}" opts=()
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[OS16] SKIP train ${name}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${prob}" "${blocks}" "${mix}" "${gray}" "${gray_scope}" "${sample}" "${donor}" "${block}" "${schedule}" "${start}" "${end}" "${period}" "${on}")
  echo "[OS16] TRAIN gpu=${gpu} exp=${name} family=${family} p=${prob} b=${blocks} m=${mix} gray=${gray}/${gray_scope} sample=${sample} donor=${donor} block=${block} schedule=${schedule}:${start}-${end}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEED}" \
    TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet_stage3local'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2"
  local name family prob blocks mix gray gray_scope sample donor block schedule start end period on
  IFS='|' read -r name family prob blocks mix gray gray_scope sample donor block schedule start end period on <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" weight="${OUTPUT_BASE}/${name}/transformer_${MAX_EPOCHS}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/${name}" opts=()
  [[ -f "${weight}" ]] || { echo "[OS16] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[OS16] SKIP eval ${name}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${prob}" "${blocks}" "${mix}" "${gray}" "${gray_scope}" "${sample}" "${donor}" "${block}" "${schedule}" "${start}" "${end}" "${period}" "${on}")
  echo "[OS16] EVAL gpu=${gpu} exp=${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${SEED}" TEST.WEIGHT "'${weight}'" \
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
  OS16_SPECS="${specs}" python - "${OUTPUT_BASE}" <<'PY'
import os, re, sys

base = sys.argv[1]
specs = [line for line in os.environ.get('OS16_SPECS', '').splitlines() if line]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?\s*([0-9.]+)%')

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

print(f"{'experiment':<25} {'family':<9} {'p':>4} {'b/m':>5} {'gray':>5} {'scope':<5} {'sample':<8} {'donor':<11} {'block':<13} {'sched':<8} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
rows = []
for spec in specs:
    name, family, prob, blocks, mix, gray, scope, sample, donor, block, schedule, start, end, period, on = spec.split('|')
    path = os.path.join(base, 'eval', name, 'test_log.txt')
    result = parse(path) if os.path.exists(path) else None
    values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
    print(f"{name:<25} {family:<9} {prob:>4} {(blocks+'/'+mix):>5} {gray:>5} {scope:<5} {sample:<8} {donor:<11} {block:<13} {schedule:<8} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")
    if result is not None:
        rows.append((result['mAP'], result['R1'], name, family))

print('\nexternal_baseline: semantic_detail seed=42 iw=0.1 mAP=84.7 R1=92.3')
if rows:
    best_map = max(rows, key=lambda row: (row[0], row[1]))
    best_r1 = max(rows, key=lambda row: (row[1], row[0]))
    print(f"best_mAP: {best_map[2]} family={best_map[3]} mAP={best_map[0]:.1f} R1={best_map[1]:.1f}")
    print(f"best_R1 : {best_r1[2]} family={best_r1[3]} mAP={best_r1[0]:.1f} R1={best_r1[1]:.1f}")
PY
}

echo "[OS16] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEED}"
exit_status=0
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  if ! run_pool train; then
    echo "[OS16] WARNING: one or more training jobs failed; continuing to evaluation and summary" >&2
    exit_status=1
  fi
fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  if ! run_pool eval; then
    echo "[OS16] WARNING: one or more evaluation jobs failed; continuing to summary" >&2
    exit_status=1
  fi
fi
summarize | tee "${OUTPUT_BASE}/summary_osbbm16.txt"
echo "[OS16] Summary saved to ${OUTPUT_BASE}/summary_osbbm16.txt"
exit "${exit_status}"
