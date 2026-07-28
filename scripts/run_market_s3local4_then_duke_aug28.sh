#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Ordered experiment queue:
#   1) migrate the matched Stage3-FCU x local-supervision 2x2 test to Market;
#   2) optionally continue with the Duke homologous-appearance 28-group scan.
# Each child script owns its output directory and summary. Results from the two
# datasets are deliberately not pooled into one ranking table.

PHASE="${PHASE:-market}"            # market / duke_aug / all
MODE="${MODE:-all}"                 # train / eval / all / summary
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
SEARCH_SEED="${SEARCH_SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"

MARKET_CONFIG="${MARKET_CONFIG:-configs/Market/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
MARKET_OUTPUT="${MARKET_OUTPUT:-./logs/Market/fdmf_s3fcu_local_factorial4_s${SEARCH_SEED}}"
MARKET_FILTER="${MARKET_FILTER:-}"

DUKE_CONFIG="${DUKE_CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
DUKE_AUG_OUTPUT="${DUKE_AUG_OUTPUT:-./logs/Duke/fdmf_homologous_aug28_s${SEARCH_SEED}}"
DUKE_AUG_FILTER="${DUKE_AUG_FILTER:-}"

case "${PHASE}" in
  market|duke_aug|all) ;;
  *) echo "PHASE must be market, duke_aug, or all" >&2; exit 2 ;;
esac
case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac

run_market_factorial() {
  echo "[OrderedQueue] Market 2x2 first: seed=${SEARCH_SEED} output=${MARKET_OUTPUT}"
  env \
    MODE="${MODE}" \
    CONFIG="${MARKET_CONFIG}" \
    OUTPUT_BASE="${MARKET_OUTPUT}" \
    SEARCH_SEED="${SEARCH_SEED}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
    TEST_BATCH="${TEST_BATCH}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    OSNET_PRETRAIN="${OSNET_PRETRAIN}" \
    EXPERIMENT_FILTER="${MARKET_FILTER}" \
    EVAL_DECOMPOSE=0 \
    bash "${SCRIPT_DIR}/run_duke_fdmf_s3fcu_local_factorial4.sh" \
      "${GPU_IDS}" "${MAX_JOBS}"
  echo "[OrderedQueue] Market summary: ${MARKET_OUTPUT}/summary.txt"
}

run_duke_augmentation() {
  echo "[OrderedQueue] Duke augmentation 28: seed=${SEARCH_SEED} output=${DUKE_AUG_OUTPUT}"
  env \
    MODE="${MODE}" \
    CONFIG="${DUKE_CONFIG}" \
    OUTPUT_BASE="${DUKE_AUG_OUTPUT}" \
    SEARCH_SEED="${SEARCH_SEED}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
    TEST_BATCH="${TEST_BATCH}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    OSNET_PRETRAIN="${OSNET_PRETRAIN}" \
    EXPERIMENT_FILTER="${DUKE_AUG_FILTER}" \
    bash "${SCRIPT_DIR}/run_duke_fdmf_homologous_aug28.sh" \
      "${GPU_IDS}" "${MAX_JOBS}"
  echo "[OrderedQueue] Duke summary: ${DUKE_AUG_OUTPUT}/summary.txt"
}

if [[ "${PHASE}" == "market" || "${PHASE}" == "all" ]]; then
  run_market_factorial
fi
if [[ "${PHASE}" == "duke_aug" || "${PHASE}" == "all" ]]; then
  run_duke_augmentation
fi
