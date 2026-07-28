#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Final cross-dataset selection between two local-supervised architectures:
#   e02: Stage2 FCU + semantic-detail local (Stage3 FCU disabled)
#   e03: Stage2/Stage3 FCU + semantic-detail local
# Duke and Market each run both configurations with the same second seed.
# Existing seed-42 outputs are included in the joint summary when available.

MODE="${MODE:-all}"                 # train / eval / all / summary
DATASETS="${DATASETS:-all}"         # duke / market / all
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
SEARCH_SEED="${SEARCH_SEED:-3407}"
REFERENCE_SEED="${REFERENCE_SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"

DUKE_CONFIG="${DUKE_CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
MARKET_CONFIG="${MARKET_CONFIG:-configs/Market/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
DUKE_OUTPUT="${DUKE_OUTPUT:-./logs/Duke/fdmf_stage3_local_final2_s${SEARCH_SEED}}"
MARKET_OUTPUT="${MARKET_OUTPUT:-./logs/Market/fdmf_stage3_local_final2_s${SEARCH_SEED}}"
DUKE_REFERENCE_OUTPUT="${DUKE_REFERENCE_OUTPUT:-./logs/Duke/fdmf_s3fcu_local_factorial4_s${REFERENCE_SEED}}"
MARKET_REFERENCE_OUTPUT="${MARKET_REFERENCE_OUTPUT:-./logs/Market/fdmf_s3fcu_local_factorial4_s${REFERENCE_SEED}}"
SUMMARY_PATH="${SUMMARY_PATH:-./logs/summary_duke_market_stage3_local_final4_s${SEARCH_SEED}.txt}"
FACTORIAL_SCRIPT="${SCRIPT_DIR}/run_duke_fdmf_s3fcu_local_factorial4.sh"
FINAL_FILTER="s3l4_e02_s3off_localon,s3l4_e03_s3on_localon"

case "${MODE}" in
  train|eval|all|summary) ;;
  *) echo "MODE must be train, eval, all, or summary" >&2; exit 2 ;;
esac
case "${DATASETS}" in
  duke|market|all) ;;
  *) echo "DATASETS must be duke, market, or all" >&2; exit 2 ;;
esac

run_dataset() {
  local dataset="$1" config="$2" output="$3" jobs="$4"
  echo "[Stage3Final4] ${dataset}: e02/e03 seed=${SEARCH_SEED} jobs=${jobs} output=${output}"
  env \
    MODE="${MODE}" \
    CONFIG="${config}" \
    OUTPUT_BASE="${output}" \
    SEARCH_SEED="${SEARCH_SEED}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
    TEST_BATCH="${TEST_BATCH}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    OSNET_PRETRAIN="${OSNET_PRETRAIN}" \
    EXPERIMENT_FILTER="${FINAL_FILTER}" \
    EVAL_DECOMPOSE=0 \
    bash "${FACTORIAL_SCRIPT}" "${GPU_IDS}" "${jobs}"
}

summarize_joint() {
  mkdir -p "$(dirname "${SUMMARY_PATH}")"
  python - \
    "${MAX_EPOCHS}" "${SEARCH_SEED}" "${REFERENCE_SEED}" \
    "${DUKE_OUTPUT}" "${MARKET_OUTPUT}" \
    "${DUKE_REFERENCE_OUTPUT}" "${MARKET_REFERENCE_OUTPUT}" \
    "${SUMMARY_PATH}" <<'PY'
import os
import re
import sys

(
    epoch,
    search_seed,
    reference_seed,
    duke_output,
    market_output,
    duke_reference,
    market_reference,
    output_path,
) = sys.argv[1:]
epoch = int(epoch)
search_seed = int(search_seed)
reference_seed = int(reference_seed)

experiments = (
    ('s3l4_e02_s3off_localon', False, 'Stage2-FCU + local'),
    ('s3l4_e03_s3on_localon', True, 'Stage2/3-FCU + local'),
)
sources = (
    ('Duke', search_seed, duke_output, 'current'),
    ('Market', search_seed, market_output, 'current'),
    ('Duke', reference_seed, duke_reference, 'reference'),
    ('Market', reference_seed, market_reference, 'reference'),
)
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

records = []
seen = set()
for dataset, seed, base, provenance in sources:
    source_key = (dataset, seed, os.path.normpath(base))
    if source_key in seen:
        continue
    seen.add(source_key)
    for name, stage3, label in experiments:
        path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
        result = parse(path)
        if result is not None:
            records.append((dataset, seed, provenance, name, stage3, label, result))

lines = [
    f"{'dataset':<8} {'seed':>5} {'source':<9} {'stage3':<7} "
    f"{'architecture':<24} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}"
]
for dataset, seed, provenance, _, stage3, label, result in records:
    lines.append(
        f"{dataset:<8} {seed:>5} {provenance:<9} {str(stage3):<7} "
        f"{label:<24} {result['mAP']:>6.1f} {result['R1']:>6.1f} "
        f"{result['R5']:>6.1f} {result['R10']:>6.1f}"
    )

by_dataset_seed = {}
for dataset, seed, _, _, stage3, _, result in records:
    by_dataset_seed.setdefault((dataset, seed), {})[stage3] = result

lines.extend(['', 'Stage3 FCU effect with local supervision (on - off):'])
lines.append(f"{'dataset':<8} {'seed':>5} {'d_mAP':>8} {'d_R1':>8}")
for (dataset, seed), states in sorted(by_dataset_seed.items()):
    if False in states and True in states:
        lines.append(
            f"{dataset:<8} {seed:>5} "
            f"{states[True]['mAP'] - states[False]['mAP']:>+8.1f} "
            f"{states[True]['R1'] - states[False]['R1']:>+8.1f}"
        )

grouped = {}
for dataset, seed, _, _, stage3, _, result in records:
    grouped.setdefault((dataset, stage3), []).append(result)

lines.extend(['', 'Available-seed means:'])
lines.append(
    f"{'dataset':<8} {'stage3':<7} {'seeds':>5} {'mean_mAP':>9} "
    f"{'mean_R1':>8}"
)
for (dataset, stage3), values in sorted(grouped.items()):
    lines.append(
        f"{dataset:<8} {str(stage3):<7} {len(values):>5} "
        f"{sum(x['mAP'] for x in values) / len(values):>9.2f} "
        f"{sum(x['R1'] for x in values) / len(values):>8.2f}"
    )

cross_dataset = {False: [], True: []}
for (dataset, stage3), values in grouped.items():
    cross_dataset[stage3].append((
        sum(x['mAP'] for x in values) / len(values),
        sum(x['R1'] for x in values) / len(values),
    ))
lines.extend(['', 'Cross-dataset mean of dataset-level means:'])
for stage3 in (False, True):
    values = cross_dataset[stage3]
    if values:
        mean_map = sum(x[0] for x in values) / len(values)
        mean_r1 = sum(x[1] for x in values) / len(values)
        lines.append(
            f"stage3={str(stage3):<5} datasets={len(values)} "
            f"mean_mAP={mean_map:.2f} mean_R1={mean_r1:.2f}"
        )

text = '\n'.join(lines)
print(text)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write(text + '\n')
print(f'[Stage3Final4] Joint summary saved to {output_path}')
PY
}

if [[ "${DATASETS}" == "all" ]]; then
  # MAX_JOBS is the total four-experiment concurrency budget. Split it across
  # the two dataset workers so MAX_JOBS=4 launches 2 Duke + 2 Market jobs.
  if [[ "${MAX_JOBS}" -ge 2 ]]; then
    duke_jobs=$(((MAX_JOBS + 1) / 2))
    market_jobs=$((MAX_JOBS / 2))
    [[ "${duke_jobs}" -le 2 ]] || duke_jobs=2
    [[ "${market_jobs}" -le 2 ]] || market_jobs=2
    failures=0
    run_dataset Duke "${DUKE_CONFIG}" "${DUKE_OUTPUT}" "${duke_jobs}" &
    duke_pid=$!
    run_dataset Market "${MARKET_CONFIG}" "${MARKET_OUTPUT}" "${market_jobs}" &
    market_pid=$!
    wait "${duke_pid}" || failures=1
    wait "${market_pid}" || failures=1
    [[ "${failures}" -eq 0 ]] || exit 1
  else
    run_dataset Duke "${DUKE_CONFIG}" "${DUKE_OUTPUT}" 1
    run_dataset Market "${MARKET_CONFIG}" "${MARKET_OUTPUT}" 1
  fi
elif [[ "${DATASETS}" == "duke" ]]; then
  dataset_jobs="${MAX_JOBS}"
  [[ "${dataset_jobs}" -le 2 ]] || dataset_jobs=2
  run_dataset Duke "${DUKE_CONFIG}" "${DUKE_OUTPUT}" "${dataset_jobs}"
else
  dataset_jobs="${MAX_JOBS}"
  [[ "${dataset_jobs}" -le 2 ]] || dataset_jobs=2
  run_dataset Market "${MARKET_CONFIG}" "${MARKET_OUTPUT}" "${dataset_jobs}"
fi
if [[ "${MODE}" == "summary" || "${MODE}" == "all" ]]; then
  summarize_joint
fi
