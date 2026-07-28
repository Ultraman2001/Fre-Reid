#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_semantic_detail20}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
EVAL_EPOCHS_CSV="${EVAL_EPOCHS_CSV:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEARCH_SEED="${SEARCH_SEED:-42}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
SUMMARY_NAME="${SUMMARY_NAME:-summary_s${SEARCH_SEED}.txt}"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a EVAL_EPOCHS <<< "${EVAL_EPOCHS_CSV}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|enabled|type|mask_stage|fg|fg_stage|detail|parts|depth|temp|prior|seed
declare -a EXPERIMENTS=(
  "sd20_nolocal_s${SEARCH_SEED}|control|False|hard|stage3|False|stage3|conv2|2|0|1.0|1.0|${SEARCH_SEED}"
  "sd20_softd2_s${SEARCH_SEED}|control|True|soft|stage3|False|stage3|conv2|2|2|1.0|1.0|${SEARCH_SEED}"

  "sd20_s3_ng_o2_s${SEARCH_SEED}|structure|True|semantic_detail|stage3|False|stage3|conv2|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s3_s3g_o2_s${SEARCH_SEED}|structure|True|semantic_detail|stage3|True|stage3|conv2|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s3_s4g_o2_s${SEARCH_SEED}|structure|True|semantic_detail|stage3|True|stage4|conv2|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s4_ng_o2_s${SEARCH_SEED}|structure|True|semantic_detail|stage4|False|stage4|conv2|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s4_s3g_o2_s${SEARCH_SEED}|structure|True|semantic_detail|stage4|True|stage3|conv2|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s4_s4g_o2_s${SEARCH_SEED}|structure|True|semantic_detail|stage4|True|stage4|conv2|2|0|1.5|1.0|${SEARCH_SEED}"

  "sd20_s3_ng_o4_s${SEARCH_SEED}|structure|True|semantic_detail|stage3|False|stage3|conv4|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s3_s3g_o4_s${SEARCH_SEED}|structure|True|semantic_detail|stage3|True|stage3|conv4|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s3_s4g_o4_s${SEARCH_SEED}|structure|True|semantic_detail|stage3|True|stage4|conv4|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s4_ng_o4_s${SEARCH_SEED}|structure|True|semantic_detail|stage4|False|stage4|conv4|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s4_s3g_o4_s${SEARCH_SEED}|structure|True|semantic_detail|stage4|True|stage3|conv4|2|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_s4_s4g_o4_s${SEARCH_SEED}|structure|True|semantic_detail|stage4|True|stage4|conv4|2|0|1.5|1.0|${SEARCH_SEED}"

  "sd20_hybrid_k3_s${SEARCH_SEED}|parts|True|semantic_detail|stage3|True|stage4|conv2|3|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_hybrid_k4_s${SEARCH_SEED}|parts|True|semantic_detail|stage3|True|stage4|conv2|4|0|1.5|1.0|${SEARCH_SEED}"
  "sd20_hybrid_t10_s${SEARCH_SEED}|temperature|True|semantic_detail|stage3|True|stage4|conv2|2|0|1.0|1.0|${SEARCH_SEED}"
  "sd20_hybrid_t20_s${SEARCH_SEED}|temperature|True|semantic_detail|stage3|True|stage4|conv2|2|0|2.0|1.0|${SEARCH_SEED}"
  "sd20_hybrid_p05_s${SEARCH_SEED}|prior|True|semantic_detail|stage3|True|stage4|conv2|2|0|1.5|0.5|${SEARCH_SEED}"
  "sd20_hybrid_p20_s${SEARCH_SEED}|prior|True|semantic_detail|stage3|True|stage4|conv2|2|0|1.5|2.0|${SEARCH_SEED}"
)

if [[ -n "${EXPERIMENT_FILTER}" ]]; then
  declare -a FILTERED=()
  IFS=',' read -r -a FILTERS <<< "${EXPERIMENT_FILTER}"
  for spec in "${EXPERIMENTS[@]}"; do
    name="${spec%%|*}"
    for pattern in "${FILTERS[@]}"; do
      if [[ "${name}" == *"${pattern}"* ]]; then FILTERED+=("${spec}"); break; fi
    done
  done
  EXPERIMENTS=("${FILTERED[@]}")
fi
[[ "${#EXPERIMENTS[@]}" -gt 0 ]] || { echo "No experiment matched EXPERIMENT_FILTER=${EXPERIMENT_FILTER}" >&2; exit 2; }

common_opts() {
  local gpu="$1" enabled="$2" local_type="$3" mask_stage="$4" foreground="$5"
  local foreground_stage="$6" detail_source="$7" parts="$8" depth="$9"
  local temperature="${10}" prior="${11}"
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
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${enabled}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TYPE "'${local_type}'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES "${parts}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT 0.3 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH "${depth}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_MASK_SOURCE "'osnet'" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_INTERACTION "'hadamard'" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE "${temperature}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE "${prior}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_MARGIN 0.15 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_BALANCE_WEIGHT 0.01 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_WEIGHT 0.01 \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_MASK_STAGE "'${mask_stage}'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE "${foreground}" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_STAGE "'${foreground_stage}'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_SOURCE "'${detail_source}'" \
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
    SOLVER.EVAL_PERIOD 20 \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_train() {
  local idx="$1" spec="$2"
  local name family enabled local_type mask_stage foreground foreground_stage detail_source parts depth temperature prior seed
  IFS='|' read -r name family enabled local_type mask_stage foreground foreground_stage detail_source parts depth temperature prior seed <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" output_dir="${OUTPUT_BASE}/${name}" feature opts=()
  feature='weighted_mamba_fdmf_osnet_stage3local'; [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then echo "[SD20] SKIP train ${name}"; return; fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${local_type}" "${mask_stage}" "${foreground}" "${foreground_stage}" "${detail_source}" "${parts}" "${depth}" "${temperature}" "${prior}")
  echo "[SD20] TRAIN gpu=${gpu} exp=${name} mask=${mask_stage} fg=${foreground}/${foreground_stage} detail=${detail_source} k=${parts} T=${temperature} prior=${prior}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${seed}" TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2" epoch="$3"
  local name family enabled local_type mask_stage foreground foreground_stage detail_source parts depth temperature prior seed
  IFS='|' read -r name family enabled local_type mask_stage foreground foreground_stage detail_source parts depth temperature prior seed <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" weight="${OUTPUT_BASE}/${name}/transformer_${epoch}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${epoch}/${name}" feature opts=()
  feature='weighted_mamba_fdmf_osnet_stage3local'; [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  [[ -f "${weight}" ]] || { echo "[SD20] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then echo "[SD20] SKIP eval ${name} ep=${epoch}"; return; fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${local_type}" "${mask_stage}" "${foreground}" "${foreground_stage}" "${detail_source}" "${parts}" "${depth}" "${temperature}" "${prior}")
  echo "[SD20] EVAL gpu=${gpu} exp=${name} ep=${epoch}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${seed}" TEST.WEIGHT "'${weight}'" TEST.FEAT_MODE "'${feature}'" \
    TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local action="$1" running=0 failures=0 idx=0 spec epoch
  for spec in "${EXPERIMENTS[@]}"; do
    if [[ "${action}" == "train" ]]; then
      run_train "${idx}" "${spec}" & idx=$((idx + 1)); running=$((running + 1))
    else
      for epoch in "${EVAL_EPOCHS[@]}"; do
        run_eval "${idx}" "${spec}" "${epoch}" & idx=$((idx + 1)); running=$((running + 1))
        if [[ "${running}" -ge "${MAX_JOBS}" ]]; then wait -n || failures=1; running=$((running - 1)); fi
      done
      continue
    fi
    if [[ "${running}" -ge "${MAX_JOBS}" ]]; then wait -n || failures=1; running=$((running - 1)); fi
  done
  while [[ "${running}" -gt 0 ]]; do wait -n || failures=1; running=$((running - 1)); done
  return "${failures}"
}

summarize() {
  local specs="$(printf '%s\n' "${EXPERIMENTS[@]}")"
  SD20_SPECS="${specs}" python - "${OUTPUT_BASE}" "${EVAL_EPOCHS_CSV}" <<'PY'
import os, re, sys
base, epochs_csv = sys.argv[1:3]
epochs = [int(x) for x in epochs_csv.split(',') if x]
specs = [x for x in os.environ.get('SD20_SPECS', '').splitlines() if x]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%'); rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')
def parse(path):
    out = {}
    with open(path, encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            match = map_re.search(line)
            if match: out = {'mAP': float(match.group(1))}; continue
            match = rank_re.search(line)
            if match and 'mAP' in out: out['R' + match.group(1)] = float(match.group(2))
    return out if all(key in out for key in ('mAP', 'R1', 'R5', 'R10')) else None
print(f"{'experiment':<34} {'fam':<11} {'mask':<6} {'fg':<9} {'detail':<6} {'k':>2} {'T':>4} {'prior':>5} {'ep':>4} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
final = []
for spec in specs:
    name, family, enabled, typ, mask, fg, fg_stage, detail, parts, depth, temp, prior, seed = spec.split('|')
    fg_label = fg_stage if fg == 'True' else 'none'
    for epoch in epochs:
        path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
        result = parse(path) if os.path.exists(path) else None
        values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
        print(f"{name:<34} {family:<11} {mask:<6} {fg_label:<9} {detail:<6} {parts:>2} {temp:>4} {prior:>5} {epoch:>4} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")
        if epoch == 160 and result is not None: final.append((result['mAP'], result['R1'], name, family))
if final:
    best_map = max(final, key=lambda row: (row[0], row[1]))
    best_r1 = max(final, key=lambda row: (row[1], row[0]))
    print(f"\nbest_mAP: {best_map[2]} family={best_map[3]} mAP={best_map[0]:.1f} R1={best_map[1]:.1f}")
    print(f"best_R1 : {best_r1[2]} family={best_r1[3]} mAP={best_r1[0]:.1f} R1={best_r1[1]:.1f}")
PY
}

echo "[SD20] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS} seed=${SEARCH_SEED}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/${SUMMARY_NAME}"
echo "[SD20] Summary saved to ${OUTPUT_BASE}/${SUMMARY_NAME}"
