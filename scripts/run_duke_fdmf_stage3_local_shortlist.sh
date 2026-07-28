#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
LOCAL24_SCRIPT="scripts/run_duke_fdmf_stage3_local24.sh"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_stage3_local24}"

echo "[LocalShortlist] Running the no-local control and five shortlisted methods with seed 3407"
SEARCH_SEED=3407 \
EXPERIMENT_FILTER="nolocal,hard_k2_d1,soft_mamba_k2_had,soft_os_k2_t15,soft_os_k2_concat,soft_os_k2_d2" \
SUMMARY_NAME="summary_shortlist_s3407.txt" \
OUTPUT_BASE="${OUTPUT_BASE}" \
bash "${LOCAL24_SCRIPT}" "${GPU_IDS}" "${MAX_JOBS}"

python - "${OUTPUT_BASE}" <<'PY' | tee "${OUTPUT_BASE}/summary_shortlist_2seed.txt"
import os
import re
import statistics
import sys

base = sys.argv[1]
configs = [
    'nolocal',
    'hard_k2_d1',
    'soft_mamba_k2_had',
    'soft_os_k2_t15',
    'soft_os_k2_concat',
    'soft_os_k2_d2',
]
seeds = (42, 3407)
epochs = (40, 80, 120, 160)
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


records = {}
print(f"{'configuration':<25} {'ep':>4} {'seed':>5} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
for config in configs:
    for epoch in epochs:
        for seed in seeds:
            name = f'l24_{config}_s{seed}'
            path = os.path.join(base, 'eval', f'ep{epoch}', name, 'test_log.txt')
            result = parse(path) if os.path.exists(path) else None
            records[(config, epoch, seed)] = result
            values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
            print(f"{config:<25} {epoch:>4} {seed:>5} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")

print('\ntwo-seed means')
summary = {}
for config in configs:
    for epoch in epochs:
        results = [records[(config, epoch, seed)] for seed in seeds]
        results = [result for result in results if result is not None]
        if not results:
            continue
        maps = [result['mAP'] for result in results]
        r1s = [result['R1'] for result in results]
        values = (
            statistics.mean(maps),
            statistics.pstdev(maps) if len(maps) > 1 else 0.0,
            statistics.mean(r1s),
            statistics.pstdev(r1s) if len(r1s) > 1 else 0.0,
            len(results),
        )
        summary[(config, epoch)] = values
        print(
            f"{config:<25} ep={epoch:>3} n={values[4]} "
            f"mAP={values[0]:.2f}+-{values[1]:.2f} R1={values[2]:.2f}+-{values[3]:.2f}"
        )

baseline = summary.get(('nolocal', 160))
if baseline:
    print('\nfinal deltas versus no-local')
    final = []
    for config in configs:
        values = summary.get((config, 160))
        if values is None:
            continue
        delta_map = values[0] - baseline[0]
        delta_r1 = values[2] - baseline[2]
        final.append((values[0], values[2], config, delta_map, delta_r1))
        print(
            f"{config:<25} mAP={values[0]:.2f} ({delta_map:+.2f}) "
            f"R1={values[2]:.2f} ({delta_r1:+.2f})"
        )
    best_map = max(final, key=lambda row: (row[0], row[1]))
    best_r1 = max(final, key=lambda row: (row[1], row[0]))
    print(f"\nbest_mAP: {best_map[2]} mean_mAP={best_map[0]:.2f} mean_R1={best_map[1]:.2f}")
    print(f"best_R1 : {best_r1[2]} mean_mAP={best_r1[0]:.2f} mean_R1={best_r1[1]:.2f}")
PY

echo "[LocalShortlist] Completed"
echo "[LocalShortlist] Seed 3407: ${OUTPUT_BASE}/summary_shortlist_s3407.txt"
echo "[LocalShortlist] Two-seed comparison: ${OUTPUT_BASE}/summary_shortlist_2seed.txt"
