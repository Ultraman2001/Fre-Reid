#!/usr/bin/env bash
# Inference-weight sweep for the two-seed best semantic-detail checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
SEED42_BASE="${SEED42_BASE:-./logs/Duke/fdmf_semantic_detail20}"
SEED3407_BASE="${SEED3407_BASE:-./logs/Duke/fdmf_semantic_detail20_seed3407}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_semantic_detail_best_infer_sweep}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
INFER_WEIGHTS_CSV="${INFER_WEIGHTS_CSV:-0.0,0.1,0.2,0.3,0.4,0.5,0.7}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a INFER_WEIGHTS <<< "${INFER_WEIGHTS_CSV}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")
mkdir -p "${OUTPUT_BASE}"

# name|seed|checkpoint
declare -a CHECKPOINTS=(
  "sd20_s3_s3g_o4_s42|42|${SEED42_BASE}/sd20_s3_s3g_o4_s42/transformer_160.pth"
  "sd20_s3_s3g_o4_s3407|3407|${SEED3407_BASE}/sd20_s3_s3g_o4_s3407/transformer_160.pth"
)

tag_float() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "${value}"
}

common_opts() {
  local gpu="$1" infer_weight="$2"
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
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${infer_weight}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 0 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE 1.5 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE 1.0 \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_MASK_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_GATE True \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_FOREGROUND_STAGE "'stage3'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_SOURCE "'conv4'" \
    MODEL.OSNET_FUSION.STAGE3_DETAIL_RESIDUAL_INJECTION False \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED False \
    MODEL.MAMBAVISION.USE_SFM False \
    SOLVER.RATR_ENABLED False \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_eval() {
  local idx="$1" spec="$2" infer_weight="$3"
  local name seed checkpoint gpu tag eval_name eval_dir test_log
  local -a opts=()
  IFS='|' read -r name seed checkpoint <<< "${spec}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  tag="$(tag_float "${infer_weight}")"
  eval_name="${name}_iw${tag}"
  eval_dir="${OUTPUT_BASE}/${eval_name}"
  test_log="${eval_dir}/test_log.txt"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[SD-Infer] MISSING ${checkpoint}"
    return
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${test_log}" ]] && grep -q "Rank-10" "${test_log}"; then
    echo "[SD-Infer] SKIP ${eval_name}"
    return
  fi

  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${infer_weight}")
  echo "[SD-Infer] EVAL gpu=${gpu} seed=${seed} iw=${infer_weight} weight=${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" \
    SOLVER.SEED "${seed}" \
    TEST.WEIGHT "'${checkpoint}'" \
    TEST.FEAT_MODE "'weighted_mamba_fdmf_osnet_stage3local'" \
    TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local running=0 failures=0 job_idx=0 spec infer_weight
  for spec in "${CHECKPOINTS[@]}"; do
    for infer_weight in "${INFER_WEIGHTS[@]}"; do
      run_eval "${job_idx}" "${spec}" "${infer_weight}" &
      job_idx=$((job_idx + 1))
      running=$((running + 1))
      if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
        wait -n || failures=1
        running=$((running - 1))
      fi
    done
  done
  while [[ "${running}" -gt 0 ]]; do
    wait -n || failures=1
    running=$((running - 1))
  done
  return "${failures}"
}

summarize() {
  local specs
  specs="$(printf '%s\n' "${CHECKPOINTS[@]}")"
  SD_INFER_SPECS="${specs}" python - "${OUTPUT_BASE}" "${INFER_WEIGHTS_CSV}" <<'PY'
import os
import re
import statistics
import sys
from collections import defaultdict

base, weights_csv = sys.argv[1:3]
weights = [value for value in weights_csv.split(',') if value]
specs = [line for line in os.environ.get('SD_INFER_SPECS', '').splitlines() if line]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')

def tag_float(value):
    return value.replace('-', 'm').replace('.', 'p')

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

rows = []
for spec in specs:
    name, seed, checkpoint = spec.split('|')
    for weight in weights:
        eval_name = f'{name}_iw{tag_float(weight)}'
        path = os.path.join(base, eval_name, 'test_log.txt')
        result = parse(path) if os.path.exists(path) else None
        rows.append((name, seed, float(weight), result))

print(f"{'experiment':<39} {'seed':>5} {'iw':>4} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
for name, seed, weight, result in rows:
    values = ['NA'] * 4 if result is None else [f"{result[key]:.1f}" for key in ('mAP', 'R1', 'R5', 'R10')]
    print(f"{name:<39} {seed:>5} {weight:>4.1f} {values[0]:>6} {values[1]:>6} {values[2]:>6} {values[3]:>6}")

groups = defaultdict(list)
for _, _, weight, result in rows:
    if result is not None:
        groups[weight].append(result)

print()
print(f"{'iw':>4} {'runs':>4} {'mean_mAP':>9} {'std':>6} {'mean_R1':>8} {'std':>6} {'mean_R5':>8} {'mean_R10':>9}")
summary = []
for weight in sorted(groups):
    results = groups[weight]
    maps = [row['mAP'] for row in results]
    r1s = [row['R1'] for row in results]
    r5s = [row['R5'] for row in results]
    r10s = [row['R10'] for row in results]
    map_std = statistics.pstdev(maps) if len(maps) > 1 else 0.0
    r1_std = statistics.pstdev(r1s) if len(r1s) > 1 else 0.0
    item = (weight, len(results), statistics.mean(maps), map_std, statistics.mean(r1s), r1_std)
    summary.append(item)
    print(f"{weight:>4.1f} {len(results):>4} {item[2]:>9.2f} {map_std:>6.2f} {item[4]:>8.2f} {r1_std:>6.2f} {statistics.mean(r5s):>8.2f} {statistics.mean(r10s):>9.2f}")

complete = [row for row in summary if row[1] == len(specs)]
if complete:
    best_map = max(complete, key=lambda row: (row[2], row[4], -row[3], -row[0]))
    best_r1 = max(complete, key=lambda row: (row[4], row[2], -row[5], -row[0]))
    print(f"\nbest_mean_mAP: iw={best_map[0]:.1f} mAP={best_map[2]:.2f}+-{best_map[3]:.2f} R1={best_map[4]:.2f}+-{best_map[5]:.2f}")
    print(f"best_mean_R1 : iw={best_r1[0]:.1f} mAP={best_r1[2]:.2f}+-{best_r1[3]:.2f} R1={best_r1[4]:.2f}+-{best_r1[5]:.2f}")
PY
}

echo "[SD-Infer] checkpoints=${#CHECKPOINTS[@]} infer_weights=${#INFER_WEIGHTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS}"
run_pool
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[SD-Infer] Summary saved to ${OUTPUT_BASE}/summary.txt"
