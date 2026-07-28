#!/usr/bin/env bash
# Cross-dataset causal validation (24 groups) plus four consensus/discrepancy
# deepening variants on Duke. Existing e18 anchors are not retrained.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Usage: bash scripts/run_fdmf_odsmf_cv28.sh 0,1 4
MODE="${MODE:-all}"  # train / eval / all / summary
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0,1}}"
MAX_JOBS="${2:-4}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
MAMBA_PRETRAIN="${MAMBA_PRETRAIN:-/workspace/pretrained/mambavision_tiny_1k.pth.tar}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac
if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer" >&2
  exit 2
fi
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|dataset|enabled|C|D|depth|packed|gate|shuffle|detach|perp|modulation
EXPERIMENTS=(
  "cv28_d_e00_legacy|baseline|duke|False|True|True|1|False|separable|False|False|False|none"
  "cv28_d_e01_noscan|scan|duke|True|True|True|0|False|separable|False|False|False|none"
  "cv28_d_e02_consensus|state|duke|True|True|False|1|False|separable|False|False|False|none"
  "cv28_d_e03_discrepancy|state|duke|True|False|True|1|False|separable|False|False|False|none"
  "cv28_d_e04_packed|topology|duke|True|True|True|1|True|separable|False|False|False|none"
  "cv28_d_e05_nogate|gate|duke|True|True|True|1|False|none|False|False|False|none"
  "cv28_d_e06_shuffle_d|causal|duke|True|True|True|1|False|separable|True|False|False|none"
  "cv28_d_e07_detach_d|gradient|duke|True|True|True|1|False|separable|False|True|False|none"

  "cv28_m_e00_legacy|baseline|market|False|True|True|1|False|separable|False|False|False|none"
  "cv28_m_e01_noscan|scan|market|True|True|True|0|False|separable|False|False|False|none"
  "cv28_m_e02_consensus|state|market|True|True|False|1|False|separable|False|False|False|none"
  "cv28_m_e03_discrepancy|state|market|True|False|True|1|False|separable|False|False|False|none"
  "cv28_m_e04_packed|topology|market|True|True|True|1|True|separable|False|False|False|none"
  "cv28_m_e05_nogate|gate|market|True|True|True|1|False|none|False|False|False|none"
  "cv28_m_e06_shuffle_d|causal|market|True|True|True|1|False|separable|True|False|False|none"
  "cv28_m_e07_detach_d|gradient|market|True|True|True|1|False|separable|False|True|False|none"

  "cv28_ms_e00_legacy|baseline|msmt|False|True|True|1|False|separable|False|False|False|none"
  "cv28_ms_e01_noscan|scan|msmt|True|True|True|0|False|separable|False|False|False|none"
  "cv28_ms_e02_consensus|state|msmt|True|True|False|1|False|separable|False|False|False|none"
  "cv28_ms_e03_discrepancy|state|msmt|True|False|True|1|False|separable|False|False|False|none"
  "cv28_ms_e04_packed|topology|msmt|True|True|True|1|True|separable|False|False|False|none"
  "cv28_ms_e05_nogate|gate|msmt|True|True|True|1|False|none|False|False|False|none"
  "cv28_ms_e06_shuffle_d|causal|msmt|True|True|True|1|False|separable|True|False|False|none"
  "cv28_ms_e07_detach_d|gradient|msmt|True|True|True|1|False|separable|False|True|False|none"

  "cv28_d_e24_semantic_perp|deepening|duke|True|True|True|1|False|separable|False|False|True|none"
  "cv28_d_e25_c2d_mod|deepening|duke|True|True|True|1|False|separable|False|False|False|c2d"
  "cv28_d_e26_d2c_reverse|reverse|duke|True|True|True|1|False|separable|False|False|False|d2c"
  "cv28_d_e27_perp_c2d_full|proposed|duke|True|True|True|1|False|separable|False|False|True|c2d"
)

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_FAMILY SPEC_DATASET SPEC_ENABLED SPEC_C SPEC_D \
    SPEC_DEPTH SPEC_PACKED SPEC_GATE SPEC_SHUFFLE SPEC_DETACH SPEC_PERP \
    SPEC_MODULATION <<< "$1"
}

set_dataset() {
  case "${SPEC_DATASET}" in
    duke)
      SPEC_DATASET_NAME="dukemtmc"
      SPEC_SEED=3407
      SPEC_CONFIG="configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml"
      SPEC_OUTPUT_ROOT="./logs/Duke/fdmf_odsmf_cv28_s3407"
      SPEC_DROP_PATH=0.5
      SPEC_SFM_DROP_PATH=0.01
      SPEC_ANCHOR_ROOT="./logs/Duke/fdmf_odsmf_e18_s3407"
      ;;
    market)
      SPEC_DATASET_NAME="market1501"
      SPEC_SEED=42
      SPEC_CONFIG="configs/Market/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml"
      SPEC_OUTPUT_ROOT="./logs/Market/fdmf_odsmf_cv28_s42"
      SPEC_DROP_PATH=0.3
      SPEC_SFM_DROP_PATH=0.01
      SPEC_ANCHOR_ROOT="./logs/Market/fdmf_odsmf_e18_s42"
      ;;
    msmt)
      SPEC_DATASET_NAME="msmt17"
      SPEC_SEED=42
      SPEC_CONFIG="configs/MSMT17/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml"
      SPEC_OUTPUT_ROOT="./logs/MSMT17/fdmf_odsmf_cv28_s42"
      SPEC_DROP_PATH=0.5
      SPEC_SFM_DROP_PATH=0.45
      SPEC_ANCHOR_ROOT="./logs/MSMT17/fdmf_odsmf_e18_s42"
      ;;
    *) echo "Unknown dataset tag: ${SPEC_DATASET}" >&2; return 2 ;;
  esac
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
  local gpu="$1" output="$2"
  printf '%s\n' \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.PRETRAIN_CHOICE "'imagenet'" \
    MODEL.PRETRAIN_PATH "'${MAMBA_PRETRAIN}'" \
    MODEL.METRIC_LOSS_TYPE "'triplet'" \
    MODEL.IF_LABELSMOOTH "'on'" \
    MODEL.IF_WITH_CENTER "'no'" \
    MODEL.NAME "'transformer'" \
    MODEL.NO_MARGIN True \
    MODEL.TRANSFORMER_TYPE "'mambavision_tiny_TransReID'" \
    MODEL.DROP_PATH "${SPEC_DROP_PATH}" \
    MODEL.DROP_OUT 0.0 \
    MODEL.ATT_DROP_RATE 0.0 \
    MODEL.ID_LOSS_WEIGHT 1.0 \
    MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
    MODEL.POOLING_TYPE "'gem'" \
    MODEL.SIE_CAMERA False \
    MODEL.SIE_XISHU 1.5 \
    MODEL.EMA.ENABLED False \
    MODEL.MAMBAVISION.SASF_STAGES "[]" \
    MODEL.MAMBAVISION.GLOBAL_STAGES "[2,3]" \
    MODEL.MAMBAVISION.USE_SFM False \
    MODEL.MAMBAVISION.SFM_DEPTHS "[1,2,3]" \
    MODEL.MAMBAVISION.SFM_DROP_PATH "${SPEC_SFM_DROP_PATH}" \
    MODEL.MAMBAVISION.USE_FINE_BRANCH False \
    MODEL.OSNET_FUSION.ENABLED True \
    MODEL.OSNET_FUSION.OSNET_TYPE "'osnet_x1_0'" \
    MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'" \
    MODEL.OSNET_FUSION.FREEZE_OSNET False \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT 0.5 \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT 1.0 \
    MODEL.OSNET_FUSION.FUSION_TYPE "'fdmf'" \
    MODEL.OSNET_FUSION.FUSION_NORM "'none'" \
    MODEL.OSNET_FUSION.FUSION_BETA 1.0 \
    MODEL.OSNET_FUSION.FCU_ENABLED True \
    MODEL.OSNET_FUSION.FCU_EXCHANGE_ENABLED True \
    MODEL.OSNET_FUSION.FCU_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FCU_STAGES "[1,2]" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE1_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "''" \
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
    MODEL.OSNET_FUSION.FDMF_MLP_RATIO 2.0 \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_FORWARD_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_INIT 0.1 \
    MODEL.OSNET_FUSION.FDMF_DESCRIPTOR_MAMBA_WEIGHT 1.0 \
    MODEL.OSNET_FUSION.FDMF_DESCRIPTOR_FDMF_WEIGHT 0.75 \
    MODEL.OSNET_FUSION.FDMF_DESCRIPTOR_OSNET_WEIGHT 0.4 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_ENABLED "${SPEC_ENABLED}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_STATE_DIM 256 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_DEPTH "${SPEC_DEPTH}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_USE_CONSENSUS "${SPEC_C}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_USE_DISCREPANCY "${SPEC_D}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHARE_MAMBA True \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHARE_NORM False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_PACKED_SCAN "${SPEC_PACKED}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_BASIS "'fixed'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_GATE "'${SPEC_GATE}'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHUFFLE_DISCREPANCY "${SPEC_SHUFFLE}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_DETACH_DISCREPANCY "${SPEC_DETACH}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_CARRIER_ENABLED False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_RES_SCALE_INIT 0.1 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_RES_SCALE_MAX 0.5 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SEMANTIC_PERP "${SPEC_PERP}" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_MODULATION "'${SPEC_MODULATION}'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_MODULATION_SCALE 0.5 \
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
    INPUT.SIZE_TRAIN "[256,128]" \
    INPUT.SIZE_TEST "[256,128]" \
    INPUT.PROB 0.5 \
    INPUT.PADDING 10 \
    INPUT.PIXEL_MEAN "[0.485,0.456,0.406]" \
    INPUT.PIXEL_STD "[0.229,0.224,0.225]" \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED False \
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
    DATASETS.NAMES "('${SPEC_DATASET_NAME}')" \
    DATASETS.ROOT_DIR "'${DATA_ROOT}'" \
    DATALOADER.SAMPLER "'softmax_triplet'" \
    DATALOADER.NUM_INSTANCE 4 \
    DATALOADER.NUM_WORKERS 8 \
    SOLVER.OPTIMIZER_NAME "'AdamW'" \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.BASE_LR 0.00035 \
    SOLVER.IMS_PER_BATCH 64 \
    SOLVER.WARMUP_METHOD "'linear'" \
    SOLVER.WARMUP_EPOCHS 20 \
    SOLVER.LARGE_FC_LR False \
    SOLVER.CHECKPOINT_PERIOD 40 \
    SOLVER.LOG_PERIOD 50 \
    SOLVER.EVAL_PERIOD "${MAX_EPOCHS}" \
    SOLVER.WEIGHT_DECAY 0.05 \
    SOLVER.WEIGHT_DECAY_BIAS 0.05 \
    SOLVER.BIAS_LR_FACTOR 1 \
    SOLVER.SEED "${SPEC_SEED}" \
    SOLVER.MARGIN 0.3 \
    SOLVER.STAGE4_LR_FACTOR 2.0 \
    SOLVER.SASF_LR_FACTOR 1.0 \
    SOLVER.OSNET_LR_FACTOR 2.0 \
    SOLVER.OSNET_WEIGHT_DECAY 0.0005 \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS 0.0005 \
    SOLVER.OSNET_FUSION_LR_FACTOR 3.0 \
    SOLVER.FDMF_LR_FACTOR 3.0 \
    SOLVER.FCU_LR_FACTOR 3.0 \
    SOLVER.CLIP_GRAD_NORM 10.0 \
    SOLVER.RATR_ENABLED False \
    TEST.EVAL True \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    TEST.RE_RANKING False \
    TEST.FEAT_CONCAT False \
    TEST.EVAL_ALL_FEATS False \
    TEST.BRANCH_NORM_BETAS "[0.25,0.5,0.63,0.75]" \
    TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" \
    OUTPUT_DIR "'${output}'"
}

run_train() {
  local gpu="$1" spec="$2" output_dir process_log
  local -a opts=()
  read_spec "${spec}"
  set_dataset || return 1
  output_dir="${SPEC_OUTPUT_ROOT}/${SPEC_NAME}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[CV28] SKIP train ${SPEC_NAME}"
    return 0
  fi
  mkdir -p "${output_dir}"
  process_log="${output_dir}/process_train.log"
  mapfile -t opts < <(common_opts "${gpu}" "${output_dir}")
  echo "[CV28] TRAIN gpu=${gpu} exp=${SPEC_NAME} data=${SPEC_DATASET} enabled=${SPEC_ENABLED} perp=${SPEC_PERP} mod=${SPEC_MODULATION}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${SPEC_CONFIG}" \
    "${opts[@]}" TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet'" \
    2>&1 | tee "${process_log}"
}

run_eval_mode() {
  local gpu="$1" weight="$2" mode="$3" eval_dir="$4" process_log
  local -a opts=()
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && \
     grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[CV28] SKIP eval ${SPEC_NAME}/${mode}"
    return 0
  fi
  mkdir -p "${eval_dir}"
  process_log="${eval_dir}/process_eval.log"
  mapfile -t opts < <(common_opts "${gpu}" "${eval_dir}")
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${SPEC_CONFIG}" \
    "${opts[@]}" TEST.WEIGHT "'${weight}'" TEST.FEAT_MODE "'${mode}'" \
    OUTPUT_DIR "'${eval_dir}'" 2>&1 | tee "${process_log}"
}

run_eval() {
  local gpu="$1" spec="$2" output_dir weight mode eval_dir
  read_spec "${spec}"
  set_dataset || return 1
  output_dir="${SPEC_OUTPUT_ROOT}/${SPEC_NAME}"
  weight="${output_dir}/transformer_${MAX_EPOCHS}.pth"
  if [[ ! -f "${weight}" ]]; then
    echo "[CV28] MISSING ${weight}" >&2
    return 1
  fi
  for mode in fdmf_only fdmf weighted_mamba_fdmf_osnet; do
    eval_dir="${SPEC_OUTPUT_ROOT}/eval/ep${MAX_EPOCHS}/${mode}/${SPEC_NAME}"
    echo "[CV28] EVAL gpu=${gpu} exp=${SPEC_NAME} mode=${mode}"
    run_eval_mode "${gpu}" "${weight}" "${mode}" "${eval_dir}" || return 1
  done
}

run_workers() {
  local action="$1"
  shift
  local -a specs=("$@") pids=()
  local worker_count="${MAX_JOBS}" slot pid failures=0
  if [[ "${worker_count}" -gt "${#specs[@]}" ]]; then
    worker_count="${#specs[@]}"
  fi
  for ((slot=0; slot<worker_count; slot++)); do
    (
      local gpu="${GPUS[$((slot % ${#GPUS[@]}))]}" idx
      for ((idx=slot; idx<${#specs[@]}; idx+=worker_count)); do
        if [[ "${action}" == "train" ]]; then
          run_train "${gpu}" "${specs[$idx]}" || \
            echo "[CV28] FAILED train ${specs[$idx]%%|*}" >&2
        else
          run_eval "${gpu}" "${specs[$idx]}" || \
            echo "[CV28] FAILED eval ${specs[$idx]%%|*}" >&2
        fi
      done
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=1
  done
  return "${failures}"
}

summarize() {
  local summary="./logs/fdmf_odsmf_cv28_summary.txt" spec mode log map r1 r5 r10
  mkdir -p ./logs
  {
    echo "Existing e18 anchors are not retrained in CV28."
    printf '%-30s %-10s %-9s %-5s %-5s %-5s %-10s %-9s %-7s %-5s %-5s %-29s %6s %6s %6s %6s\n' \
      experiment dataset family C/D dep pack gate shuffle detach perp mod mode mAP R1 R5 R10
    for SPEC_DATASET in duke market msmt; do
      set_dataset || continue
      case "${SPEC_DATASET}" in
        duke) anchor_name="anchor_e18_duke_s3407" ;;
        market) anchor_name="anchor_e18_market_s42" ;;
        msmt) anchor_name="anchor_e18_msmt_s42" ;;
      esac
      for mode in fdmf_only fdmf weighted_mamba_fdmf_osnet; do
        log="${SPEC_ANCHOR_ROOT}/eval/ep${MAX_EPOCHS}/${mode}/test_log.txt"
        [[ -f "${log}" ]] || continue
        map="$(grep -E 'mAP:' "${log}" | tail -n 1 | sed -E 's/.*mAP:[[:space:]]*([0-9.]+)%.*/\1/')"
        r1="$(grep -E 'Rank-1[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        r5="$(grep -E 'Rank-5[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        r10="$(grep -E 'Rank-10[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        printf '%-30s %-10s %-9s %-5s %-5s %-5s %-10s %-9s %-7s %-5s %-5s %-29s %6s %6s %6s %6s\n' \
          "${anchor_name}" "${SPEC_DATASET}" anchor CD 1 False separable False False False none "${mode}" \
          "${map:--}" "${r1:--}" "${r5:--}" "${r10:--}"
      done
    done
    for spec in "${SELECTED[@]}"; do
      read_spec "${spec}"
      set_dataset || continue
      for mode in fdmf_only fdmf weighted_mamba_fdmf_osnet; do
        log="${SPEC_OUTPUT_ROOT}/eval/ep${MAX_EPOCHS}/${mode}/${SPEC_NAME}/test_log.txt"
        [[ -f "${log}" ]] || continue
        map="$(grep -E 'mAP:' "${log}" | tail -n 1 | sed -E 's/.*mAP:[[:space:]]*([0-9.]+)%.*/\1/')"
        r1="$(grep -E 'Rank-1[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        r5="$(grep -E 'Rank-5[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        r10="$(grep -E 'Rank-10[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        states="$( [[ "${SPEC_C}" == True ]] && printf '%s' C || printf '%s' - )$( [[ "${SPEC_D}" == True ]] && printf '%s' D || printf '%s' - )"
        printf '%-30s %-10s %-9s %-5s %-5s %-5s %-10s %-9s %-7s %-5s %-5s %-29s %6s %6s %6s %6s\n' \
          "${SPEC_NAME}" "${SPEC_DATASET}" "${SPEC_FAMILY}" "${states}" \
          "${SPEC_DEPTH}" "${SPEC_PACKED}" "${SPEC_GATE}" "${SPEC_SHUFFLE}" \
          "${SPEC_DETACH}" "${SPEC_PERP}" "${SPEC_MODULATION}" "${mode}" \
          "${map:--}" "${r1:--}" "${r5:--}" "${r10:--}"
      done
    done
  } | tee "${summary}"
  echo "[CV28] Summary saved to ${summary}"
}

mapfile -t SELECTED < <(select_specs)
if [[ "${#SELECTED[@]}" -eq 0 ]]; then
  echo "[CV28] No experiments selected." >&2
  exit 2
fi
echo "[CV28] Scheduler: experiments=${#SELECTED[@]}, jobs=${MAX_JOBS}, GPUs=${GPUS[*]}; failures do not block later jobs."

case "${MODE}" in
  train) run_workers train "${SELECTED[@]}" || true ;;
  eval) run_workers eval "${SELECTED[@]}" || true ;;
  all)
    run_workers train "${SELECTED[@]}" || true
    run_workers eval "${SELECTED[@]}" || true
    summarize
    ;;
  summary) summarize ;;
esac
