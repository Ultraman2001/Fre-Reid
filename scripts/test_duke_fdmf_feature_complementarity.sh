#!/usr/bin/env bash
# Analyze existing Duke checkpoints without retraining.
#
# Checkpoints:
#   - no Stage3 local, fusion LR factor 3.0, seeds 42/3407
#   - Hard Stage3 local, fusion LR factor 3.0, seeds 42/3407
#
# Usage:
#   bash scripts/test_duke_fdmf_feature_complementarity.sh 0,1 2
#   bash scripts/test_duke_fdmf_feature_complementarity.sh eval 0,1 2
#   bash scripts/test_duke_fdmf_feature_complementarity.sh summary
#   WEIGHT_SWEEP=1 bash scripts/test_duke_fdmf_feature_complementarity.sh 0,1 2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="eval"
if [[ "${1:-}" == "eval" || "${1:-}" == "summary" ]]; then
  MODE="$1"
  shift
fi

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_feature_complementarity}"
CONTROL_BASE="${CONTROL_BASE:-./logs/Duke/osnet_fdmf_s3stripe_control4}"
HARD_BASE="${HARD_BASE:-./logs/Duke/osnet_fdmf_s3stripe_ocr_ablation}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
WEIGHT_EPOCH="${WEIGHT_EPOCH:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
WEIGHT_SWEEP="${WEIGHT_SWEEP:-0}"
if [[ "${WEIGHT_SWEEP}" == "1" ]]; then
  REPORT_NAME="${REPORT_NAME:-feature_weight_sweep.json}"
else
  REPORT_NAME="${REPORT_NAME:-feature_complementarity.json}"
fi
FDMF_WEIGHT_LIST="${FDMF_WEIGHT_LIST:-[0.5,0.75,1.0]}"
OSNET_WEIGHT_LIST="${OSNET_WEIGHT_LIST:-[0.0,0.2,0.4,0.6]}"
# The first stage fixes Local=0. Run a second focused sweep after selecting F/O.
LOCAL_WEIGHT_LIST="${LOCAL_WEIGHT_LIST:-[0.0]}"

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|variant|seed|local_enabled|fusion_lr_factor|weight_path
declare -a EXPERIMENTS=(
  "no_local_s42|no_local|42|False|3.0|${CONTROL_BASE}/no_s3local_lrf3_s42/transformer_${WEIGHT_EPOCH}.pth"
  "no_local_s3407|no_local|3407|False|3.0|${CONTROL_BASE}/no_s3local_lrf3_s3407/transformer_${WEIGHT_EPOCH}.pth"
  "hard_s42|hard|42|True|3.0|${HARD_BASE}/s3ocr_hard_lw0p1_s42/transformer_${WEIGHT_EPOCH}.pth"
  "hard_s3407|hard|3407|True|3.0|${HARD_BASE}/s3ocr_hard_lw0p1_s3407/transformer_${WEIGHT_EPOCH}.pth"
)

common_opts() {
  local gpu="$1"
  local local_enabled="$2"
  local fusion_lr_factor="$3"
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
    MODEL.OSNET_FUSION.FCU_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_D_STATE 8 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_D_CONV 3 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MLP_RATIO 2.0 \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${local_enabled}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_D_STATE 8 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_D_CONV 3 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MLP_RATIO 2.0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS True \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT 0.5 \
    SOLVER.OSNET_FUSION_LR_FACTOR "${fusion_lr_factor}" \
    "${pretrain_opts[@]}"
}

run_eval() {
  local idx="$1"
  local spec="$2"
  local name variant seed local_enabled fusion_lr_factor weight_path
  IFS='|' read -r name variant seed local_enabled fusion_lr_factor weight_path <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"
  local report_path="${output_dir}/${REPORT_NAME}"
  local -a opts=()
  local -a sweep_opts=()

  if [[ ! -f "${weight_path}" ]]; then
    echo "[FDMFComplementarity] MISSING ${name}: ${weight_path}"
    return 0
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${report_path}" ]]; then
    echo "[FDMFComplementarity] SKIP ${name}: ${report_path}"
    return 0
  fi

  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${local_enabled}" "${fusion_lr_factor}")
  if [[ "${WEIGHT_SWEEP}" == "1" ]]; then
    sweep_opts=(
      TEST.COMPLEMENTARITY_WEIGHT_SWEEP True
      TEST.COMPLEMENTARITY_FDMF_WEIGHTS "${FDMF_WEIGHT_LIST}"
      TEST.COMPLEMENTARITY_OSNET_WEIGHTS "${OSNET_WEIGHT_LIST}"
      TEST.COMPLEMENTARITY_LOCAL_WEIGHTS "${LOCAL_WEIGHT_LIST}"
    )
  fi
  echo "[FDMFComplementarity] EVAL gpu=${gpu} exp=${name} variant=${variant} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" \
    SOLVER.SEED "${seed}" \
    TEST.WEIGHT "'${weight_path}'" \
    TEST.NECK_FEAT "'before'" \
    TEST.FEAT_NORM "'yes'" \
    TEST.FEAT_MODE "'mamba_fdmf_osnet'" \
    TEST.EVAL_ALL_FEATS False \
    TEST.BRANCH_NORM_BETAS "[]" \
    TEST.COMPLEMENTARITY_ANALYSIS True \
    TEST.COMPLEMENTARITY_TOPK 10 \
    TEST.COMPLEMENTARITY_CKA_SAMPLES 2048 \
    TEST.COMPLEMENTARITY_QUERY_SAMPLES 256 \
    TEST.COMPLEMENTARITY_GALLERY_SAMPLES 4096 \
    TEST.COMPLEMENTARITY_OUTPUT "'${REPORT_NAME}'" \
    "${sweep_opts[@]}" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    OUTPUT_DIR "${output_dir}"
}

run_parallel() {
  local running=0
  local idx
  for idx in "${!EXPERIMENTS[@]}"; do
    run_eval "${idx}" "${EXPERIMENTS[$idx]}" &
    running=$((running + 1))
    if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

print_summary() {
  python - "${OUTPUT_BASE}" "${REPORT_NAME}" "${WEIGHT_SWEEP}" <<'PY'
import json
import os
import sys
from collections import defaultdict

output_base = sys.argv[1]
report_name = sys.argv[2]
weight_sweep = sys.argv[3] == '1'
experiments = [
    ('no_local_s42', 'no_local', '42'),
    ('no_local_s3407', 'no_local', '3407'),
    ('hard_s42', 'hard', '42'),
    ('hard_s3407', 'hard', '3407'),
]
descriptor_order = [
    'bnorm_mamba',
    'bnorm_fdmf',
    'bnorm_osnet',
    'bnorm_mamba_osnet',
    'bnorm_mamba_fdmf',
    'bnorm_fdmf_osnet',
    'bnorm_mamba_fdmf_osnet',
    'bnorm_stage3local',
    'bnorm_mamba_fdmf_osnet_stage3local',
]

reports = []
print()
print(f"{'experiment':<20} {'variant':<10} {'seed':<6} {'descriptor':<42} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
for name, variant, seed in experiments:
    path = os.path.join(output_base, name, report_name)
    if not os.path.exists(path):
        print(f"{name:<20} {variant:<10} {seed:<6} {'missing':<42}")
        continue
    with open(path, 'r', encoding='utf-8') as handle:
        report = json.load(handle)
    reports.append((name, variant, seed, report))
    retrieval = report.get('retrieval', {})
    for descriptor in descriptor_order:
        if descriptor not in retrieval:
            continue
        row = retrieval[descriptor]
        print(
            f"{name:<20} {variant:<10} {seed:<6} {descriptor:<42} "
            f"{100 * row['mAP']:>6.2f} {100 * row['R1']:>6.2f} "
            f"{100 * row['R5']:>6.2f} {100 * row['R10']:>6.2f}"
        )

print()
print(f"{'experiment':<20} {'pair':<28} {'CKA':>8} {'dist_rho':>10} {'top10_ov':>10}")
for name, _, _, report in reports:
    for pair, row in report.get('pairwise', {}).items():
        print(
            f"{name:<20} {pair:<28} {row.get('linear_cka', 0):>8.4f} "
            f"{row.get('distance_spearman', 0):>10.4f} {row.get('topk_overlap', 0):>10.4f}"
        )

print()
print(f"{'experiment':<20} {'marginal':<40} {'d_mAP_pp':>10} {'d_R1_pp':>10}")
for name, _, _, report in reports:
    for marginal, row in report.get('marginal_gain', {}).items():
        print(
            f"{name:<20} {marginal:<40} "
            f"{100 * row['delta_mAP']:>+10.2f} {100 * row['delta_R1']:>+10.2f}"
        )

aggregates = defaultdict(list)
for _, variant, _, report in reports:
    for descriptor, row in report.get('retrieval', {}).items():
        if descriptor in descriptor_order:
            aggregates[(variant, descriptor)].append(row)

print()
print(f"{'variant':<10} {'descriptor':<42} {'runs':>5} {'mean_mAP':>10} {'mean_R1':>9}")
for (variant, descriptor), rows in sorted(aggregates.items()):
    mean_map = sum(row['mAP'] for row in rows) / len(rows)
    mean_r1 = sum(row['R1'] for row in rows) / len(rows)
    print(
        f"{variant:<10} {descriptor:<42} {len(rows):>5} "
        f"{100 * mean_map:>10.2f} {100 * mean_r1:>9.2f}"
    )

if weight_sweep:
    print()
    print('=== PER-EXPERIMENT WEIGHT-SWEEP BEST ===')
    print(f"{'experiment':<20} {'metric':<6} {'weights':<28} {'mAP':>7} {'R1':>7}")
    for name, _, _, report in reports:
        weighted = report.get('weight_sweep', {})
        if not weighted:
            continue
        best_map_name, best_map = max(weighted.items(), key=lambda item: item[1]['mAP'])
        best_r1_name, best_r1 = max(weighted.items(), key=lambda item: item[1]['R1'])
        print(
            f"{name:<20} {'mAP':<6} {best_map_name:<28} "
            f"{100 * best_map['mAP']:>7.2f} {100 * best_map['R1']:>7.2f}"
        )
        print(
            f"{name:<20} {'R1':<6} {best_r1_name:<28} "
            f"{100 * best_r1['mAP']:>7.2f} {100 * best_r1['R1']:>7.2f}"
        )

    weighted_groups = defaultdict(list)
    for _, variant, _, report in reports:
        for key, row in report.get('weight_sweep', {}).items():
            weighted_groups[(variant, key)].append(row)

    aggregate_rows = []
    for (variant, key), rows in weighted_groups.items():
        aggregate_rows.append({
            'variant': variant,
            'key': key,
            'runs': len(rows),
            'mAP': sum(row['mAP'] for row in rows) / len(rows),
            'R1': sum(row['R1'] for row in rows) / len(rows),
        })

    for variant in sorted({row['variant'] for row in aggregate_rows}):
        variant_rows = [row for row in aggregate_rows if row['variant'] == variant]
        for metric in ('mAP', 'R1'):
            print()
            print(f"=== {variant} TWO-SEED WEIGHT-SWEEP TOP BY {metric} ===")
            print(f"{'weights':<28} {'runs':>5} {'mean_mAP':>10} {'mean_R1':>9}")
            ranked = sorted(variant_rows, key=lambda row: row[metric], reverse=True)
            for row in ranked[:12]:
                print(
                    f"{row['key']:<28} {row['runs']:>5} "
                    f"{100 * row['mAP']:>10.2f} {100 * row['R1']:>9.2f}"
                )
PY
}

if [[ "${MODE}" == "eval" ]]; then
  run_parallel
fi
print_summary
