#!/usr/bin/env bash
# Final ODSMF/e18 validation:
#   GPU 0: MSMT17, seed 42
#   GPU 1: Duke, seed 3407 + Market, seed 42 (concurrent processes)
#
# Only three e18 experiments are trained. The legacy FDMF baselines are not
# retrained here; compare the generated summaries with the existing records.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"  # train / eval / all / summary
GPU_MSMT="${GPU_MSMT:-0}"
GPU_DUKE_MARKET="${GPU_DUKE_MARKET:-1}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
MAMBA_PRETRAIN="${MAMBA_PRETRAIN:-/workspace/pretrained/mambavision_tiny_1k.pth.tar}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac

# name|dataset|seed|gpu|config|output|drop_path|sfm_drop_path
DUKE_SPEC="final_e18_duke_s3407|dukemtmc|3407|${GPU_DUKE_MARKET}|configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml|./logs/Duke/fdmf_odsmf_e18_s3407|0.5|0.01"
MARKET_SPEC="final_e18_market_s42|market1501|42|${GPU_DUKE_MARKET}|configs/Market/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml|./logs/Market/fdmf_odsmf_e18_s42|0.3|0.01"
MSMT_SPEC="final_e18_msmt17_s42|msmt17|42|${GPU_MSMT}|configs/MSMT17/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml|./logs/MSMT17/fdmf_odsmf_e18_s42|0.5|0.45"
ALL_SPECS=("${DUKE_SPEC}" "${MARKET_SPEC}" "${MSMT_SPEC}")

read_spec() {
  IFS='|' read -r SPEC_NAME SPEC_DATASET SPEC_SEED SPEC_GPU SPEC_CONFIG \
    SPEC_OUTPUT SPEC_DROP_PATH SPEC_SFM_DROP_PATH <<< "$1"
}

common_opts() {
  local gpu="$1" dataset="$2" seed="$3" output="$4"
  local drop_path="$5" sfm_drop_path="$6"
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
    MODEL.DROP_PATH "${drop_path}" \
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
    MODEL.MAMBAVISION.SFM_DROP_PATH "${sfm_drop_path}" \
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
    MODEL.OSNET_FUSION.FDMF_ODSMF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_ODSMF_STATE_DIM 256 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_USE_CONSENSUS True \
    MODEL.OSNET_FUSION.FDMF_ODSMF_USE_DISCREPANCY True \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHARE_MAMBA True \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHARE_NORM False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_PACKED_SCAN False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_BASIS "'fixed'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_GATE "'separable'" \
    MODEL.OSNET_FUSION.FDMF_ODSMF_SHUFFLE_DISCREPANCY False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_DETACH_DISCREPANCY False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_CARRIER_ENABLED False \
    MODEL.OSNET_FUSION.FDMF_ODSMF_RES_SCALE_INIT 0.1 \
    MODEL.OSNET_FUSION.FDMF_ODSMF_RES_SCALE_MAX 0.5 \
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
    DATASETS.NAMES "('${dataset}')" \
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
    SOLVER.SEED "${seed}" \
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

checkpoint_path() {
  printf '%s/transformer_%s.pth' "${SPEC_OUTPUT}" "${MAX_EPOCHS}"
}

run_train() {
  local spec="$1" process_log
  local -a opts=()
  read_spec "${spec}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "$(checkpoint_path)" ]]; then
    echo "[BEST3] SKIP train ${SPEC_NAME}"
    return 0
  fi
  mkdir -p "${SPEC_OUTPUT}"
  process_log="${SPEC_OUTPUT}/process_train.log"
  mapfile -t opts < <(common_opts "${SPEC_GPU}" "${SPEC_DATASET}" "${SPEC_SEED}" \
    "${SPEC_OUTPUT}" "${SPEC_DROP_PATH}" "${SPEC_SFM_DROP_PATH}")
  echo "[BEST3] TRAIN gpu=${SPEC_GPU} exp=${SPEC_NAME} dataset=${SPEC_DATASET} seed=${SPEC_SEED}"
  CUDA_VISIBLE_DEVICES="${SPEC_GPU}" python train.py --config_file "${SPEC_CONFIG}" \
    "${opts[@]}" TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet'" \
    2>&1 | tee "${process_log}"
}

run_eval_mode() {
  local mode="$1" weight="$2" eval_dir process_log
  local -a opts=()
  eval_dir="${SPEC_OUTPUT}/eval/ep${MAX_EPOCHS}/${mode}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && \
     grep -q "Rank-10" "${eval_dir}/test_log.txt"; then
    echo "[BEST3] SKIP eval ${SPEC_NAME}/${mode}"
    return 0
  fi
  mkdir -p "${eval_dir}"
  process_log="${eval_dir}/process_eval.log"
  mapfile -t opts < <(common_opts "${SPEC_GPU}" "${SPEC_DATASET}" "${SPEC_SEED}" \
    "${eval_dir}" "${SPEC_DROP_PATH}" "${SPEC_SFM_DROP_PATH}")
  echo "[BEST3] EVAL gpu=${SPEC_GPU} exp=${SPEC_NAME} mode=${mode}"
  CUDA_VISIBLE_DEVICES="${SPEC_GPU}" python test.py --config_file "${SPEC_CONFIG}" \
    "${opts[@]}" TEST.WEIGHT "'${weight}'" TEST.FEAT_MODE "'${mode}'" \
    OUTPUT_DIR "'${eval_dir}'" 2>&1 | tee "${process_log}"
}

run_eval() {
  local spec="$1" weight mode
  read_spec "${spec}"
  weight="$(checkpoint_path)"
  if [[ ! -f "${weight}" ]]; then
    echo "[BEST3] MISSING checkpoint for ${SPEC_NAME}: ${weight}" >&2
    return 1
  fi
  for mode in fdmf_only fdmf weighted_mamba_fdmf_osnet; do
    run_eval_mode "${mode}" "${weight}" || return 1
  done
}

run_job() {
  local spec="$1"
  case "${MODE}" in
    train) run_train "${spec}" ;;
    eval) run_eval "${spec}" ;;
    all)
      run_train "${spec}" || return 1
      run_eval "${spec}"
      ;;
    summary) return 0 ;;
  esac
}

summarize() {
  local summary="./logs/fdmf_odsmf_best3_summary.txt" spec mode log map r1 r5 r10
  mkdir -p ./logs
  {
    echo "ODSMF/e18 final validation (legacy FDMF baselines are historical and not retrained)"
    printf '%-28s %-10s %6s %-29s %6s %6s %6s %6s\n' \
      experiment dataset seed mode mAP R1 R5 R10
    for spec in "${ALL_SPECS[@]}"; do
      read_spec "${spec}"
      for mode in fdmf_only fdmf weighted_mamba_fdmf_osnet; do
        log="${SPEC_OUTPUT}/eval/ep${MAX_EPOCHS}/${mode}/test_log.txt"
        [[ -f "${log}" ]] || continue
        map="$(grep -E 'mAP:' "${log}" | tail -n 1 | sed -E 's/.*mAP:[[:space:]]*([0-9.]+)%.*/\1/')"
        r1="$(grep -E 'Rank-1[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        r5="$(grep -E 'Rank-5[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        r10="$(grep -E 'Rank-10[[:space:]]*:' "${log}" | tail -n 1 | sed -E 's/.*:[[:space:]]*([0-9.]+)%.*/\1/')"
        printf '%-28s %-10s %6s %-29s %6s %6s %6s %6s\n' \
          "${SPEC_NAME}" "${SPEC_DATASET}" "${SPEC_SEED}" "${mode}" \
          "${map:--}" "${r1:--}" "${r5:--}" "${r10:--}"
      done
    done
  } | tee "${summary}"
  echo "[BEST3] Summary saved to ${summary}"
}

if [[ "${MODE}" == "summary" ]]; then
  summarize
  exit 0
fi

echo "[BEST3] Scheduler: three concurrent jobs; MSMT17 -> GPU ${GPU_MSMT}; Duke + Market -> GPU ${GPU_DUKE_MARKET}."
run_job "${MSMT_SPEC}" &
pid_msmt=$!
run_job "${DUKE_SPEC}" &
pid_duke=$!
run_job "${MARKET_SPEC}" &
pid_market=$!

failures=0
wait "${pid_msmt}" || failures=1
wait "${pid_duke}" || failures=1
wait "${pid_market}" || failures=1
summarize
exit "${failures}"
