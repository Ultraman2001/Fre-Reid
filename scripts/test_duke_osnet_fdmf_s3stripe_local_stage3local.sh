#!/usr/bin/env bash
# Re-evaluate trained Stage3 stripe-local runs with Mamba + FDMF + OSNet + stripe descriptor.
# Usage:
#   bash scripts/test_duke_osnet_fdmf_s3stripe_local_stage3local.sh 0,1 2
#   bash scripts/test_duke_osnet_fdmf_s3stripe_local_stage3local.sh 0,1 2 ./logs/Duke/osnet_fdmf_s3stripe_local
#   INFER_WEIGHT=0.2 bash scripts/test_duke_osnet_fdmf_s3stripe_local_stage3local.sh 0,1 2
#   bash scripts/test_duke_osnet_fdmf_s3stripe_local_stage3local.sh summary ./logs/Duke/osnet_fdmf_s3stripe_local
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ "${1:-}" = "--summary-only" ] || [ "${1:-}" = "summary" ]; then
  SUMMARY_ONLY=1
  GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
  MAX_JOBS=1
  OUTPUT_BASE_ARG="${2:-}"
else
  SUMMARY_ONLY=0
  GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
  MAX_JOBS="${2:-1}"
  OUTPUT_BASE_ARG="${3:-}"
fi

CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_ARG:-./logs/Duke/osnet_fdmf_s3stripe_local}}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
WEIGHT_EPOCH="${WEIGHT_EPOCH:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
INFER_WEIGHT="${INFER_WEIGHT:-0.3}"
INFER_TAG="$(printf "%03d" "$(python -c "print(round(float('${INFER_WEIGHT}') * 100))")")"
EVAL_BASE="${EVAL_BASE:-${OUTPUT_BASE}/eval_mamba_fdmf_osnet_stage3local_w${INFER_TAG}}"
SUMMARY_TSV="${EVAL_BASE}/summary.tsv"
SUMMARY_CSV="${EVAL_BASE}/summary.csv"

mkdir -p "${EVAL_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

# name|num_stripes|loss_weight|mamba_depth|share_params|part_dim
declare -a EXPERIMENTS=(
  "s3psf_k2_w01_d1_shared|2|0.1|1|True|0"
  "s3psf_k2_w02_d1_shared|2|0.2|1|True|0"
  "s3psf_k2_w03_d1_shared|2|0.3|1|True|0"
  "s3psf_k4_w01_d1_shared|4|0.1|1|True|0"
  "s3psf_k4_w02_d1_shared|4|0.2|1|True|0"
  "s3psf_k4_w03_d1_shared|4|0.3|1|True|0"
  "s3psf_k8_w01_d1_shared|8|0.1|1|True|0"
  "s3psf_k8_w02_d1_shared|8|0.2|1|True|0"
  "s3psf_k8_w03_d1_shared|8|0.3|1|True|0"
  "s3psf_k4_w02_d0_shared|4|0.2|0|True|0"
  "s3psf_k4_w02_d2_shared|4|0.2|2|True|0"
  "s3psf_k8_w02_d0_shared|8|0.2|0|True|0"
  "s3psf_k8_w02_d2_shared|8|0.2|2|True|0"
  "s3psf_k2_w02_d1_indep|2|0.2|1|False|0"
  "s3psf_k4_w02_d1_indep|4|0.2|1|False|0"
  "s3psf_k8_w02_d1_indep|8|0.2|1|False|0"
  "s3psf_k2_w02_d1_p128|2|0.2|1|True|128"
  "s3psf_k4_w02_d1_p128|4|0.2|1|True|128"
  "s3psf_k8_w02_d1_p128|8|0.2|1|True|128"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name num_stripes loss_weight mamba_depth share_params part_dim

  IFS='|' read -r name num_stripes loss_weight mamba_depth share_params part_dim <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local train_dir="${OUTPUT_BASE}/${name}"
  local weight_path="${train_dir}/transformer_${WEIGHT_EPOCH}.pth"
  local eval_dir="${EVAL_BASE}/${name}"
  local test_log="${eval_dir}/test_log.txt"
  local osnet_pretrain_opts=()

  if [ ! -f "${weight_path}" ]; then
    echo "[DukeS3PSF-TestStage3Local] Missing weight EXP=${name} weight=${weight_path}"
    return 0
  fi

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${test_log}" ] && grep -q "MAMBA_FDMF_OSNET_STAGE3LOCAL Results" "${test_log}"; then
    echo "[DukeS3PSF-TestStage3Local] SKIP completed EXP=${name} log=${test_log}"
    return 0
  fi

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi

  echo "[DukeS3PSF-TestStage3Local] GPU=${gpu} EXP=${name} weight=${weight_path} eval=${eval_dir} k=${num_stripes} loss_w=${loss_weight} depth=${mamba_depth} share=${share_params} part_dim=${part_dim} infer_w=${INFER_WEIGHT}"

  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
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
    MODEL.OSNET_FUSION.FDMF_FILTER_TYPE "'none'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_STRIPE_DEPTH 0 \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED True \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES "${num_stripes}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT "${loss_weight}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH "${mamba_depth}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS "${share_params}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM "${part_dim}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${INFER_WEIGHT}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_SUPERVISION_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LG_ENHANCE_ENABLED False \
    TEST.WEIGHT "'${weight_path}'" \
    TEST.FEAT_MODE "'mamba_fdmf_osnet_stage3local'" \
    TEST.EVAL_ALL_FEATS False \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    "${osnet_pretrain_opts[@]}" \
    OUTPUT_DIR "${eval_dir}"
}

summarize_results() {
  local specs
  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[DukeS3PSF-TestStage3Local] Summarizing results -> ${SUMMARY_TSV} and ${SUMMARY_CSV}"

  DUKE_S3PSF_STAGE3LOCAL_SPECS="${specs}" python - "${OUTPUT_BASE}" "${EVAL_BASE}" "${SUMMARY_TSV}" "${SUMMARY_CSV}" "${WEIGHT_EPOCH}" <<'PY'
import csv
import os
import re
import sys

output_base, eval_base, summary_tsv, summary_csv, weight_epoch = sys.argv[1:6]
specs = [line for line in os.environ.get("DUKE_S3PSF_STAGE3LOCAL_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "k", "loss_weight", "depth", "share", "part_dim",
    "weight_file", "test_log",
    "last_epoch", "last_mAP", "last_R1", "last_R5", "last_R10",
    "best_epoch", "best_mAP", "best_R1", "best_R5", "best_R10",
]

epoch_re = re.compile(r"Validation \(Regular\) Results - Epoch:\s*(\d+)")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank_re = re.compile(r"Rank-(1|5|10)\s*:?\s*([0-9.]+)%")
block_re = re.compile(r"(MAMBA_FDMF_OSNET_STAGE3LOCAL Results|Validation .*Results|=== .*Results ===)")

def complete(record):
    return all(record.get(k) for k in ("mAP", "R1", "R5", "R10"))

def parse_records(log_path):
    records = []
    current = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = epoch_re.search(line)
            if epoch_match:
                if current and complete(current):
                    records.append(current)
                current = {"epoch": epoch_match.group(1)}
                continue
            if current is None:
                if block_re.search(line):
                    current = {"epoch": "NA"}
                else:
                    continue
            map_match = map_re.search(line)
            if map_match:
                current["mAP"] = map_match.group(1)
                continue
            rank_match = rank_re.search(line)
            if rank_match:
                rank, value = rank_match.groups()
                current[f"R{rank}"] = value
                if rank == "10" and complete(current):
                    records.append(current)
                    current = None
    if current and complete(current):
        records.append(current)
    return records

def empty_metrics(status, weight_file, test_log):
    return {
        "status": status,
        "weight_file": weight_file,
        "test_log": test_log,
        "last_epoch": "NA",
        "last_mAP": "NA",
        "last_R1": "NA",
        "last_R5": "NA",
        "last_R10": "NA",
        "best_epoch": "NA",
        "best_mAP": "NA",
        "best_R1": "NA",
        "best_R5": "NA",
        "best_R10": "NA",
    }

rows = []
for spec in specs:
    name, k, loss_weight, depth, share, part_dim = spec.split("|")
    weight_file = os.path.join(output_base, name, f"transformer_{weight_epoch}.pth")
    test_log = os.path.join(eval_base, name, "test_log.txt")
    base = {
        "name": name,
        "k": k,
        "loss_weight": loss_weight,
        "depth": depth,
        "share": share,
        "part_dim": part_dim,
    }
    if not os.path.exists(weight_file):
        rows.append({**base, **empty_metrics("missing_weight", weight_file, test_log)})
        continue
    if not os.path.exists(test_log):
        rows.append({**base, **empty_metrics("missing_log", weight_file, test_log)})
        continue
    records = parse_records(test_log)
    if not records:
        rows.append({**base, **empty_metrics("no_eval_found", weight_file, test_log)})
        continue
    last = records[-1]
    best = max(records, key=lambda item: float(item["mAP"]))
    rows.append({
        **base,
        "status": "ok",
        "weight_file": weight_file,
        "test_log": test_log,
        "last_epoch": last["epoch"],
        "last_mAP": last["mAP"],
        "last_R1": last["R1"],
        "last_R5": last["R5"],
        "last_R10": last["R10"],
        "best_epoch": best["epoch"],
        "best_mAP": best["mAP"],
        "best_R1": best["R1"],
        "best_R5": best["R5"],
        "best_R10": best["R10"],
    })

for path, dialect in ((summary_tsv, "excel-tab"), (summary_csv, "excel")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect=dialect)
        writer.writeheader()
        writer.writerows(rows)

print()
print(
    f"{'name':<30} {'status':<14} {'k':<4} {'w':<5} {'dep':<5} {'share':<7} {'pdim':<6} "
    f"{'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} {'best_ep':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<30} {row['status']:<14} {row['k']:<4} {row['loss_weight']:<5} "
        f"{row['depth']:<5} {row['share']:<7} {row['part_dim']:<6} {row['last_epoch']:<8} "
        f"{row['last_mAP']:<8} {row['last_R1']:<8} {row['best_epoch']:<8} "
        f"{row['best_mAP']:<8} {row['best_R1']:<8}"
    )
PY
}

if [ "${SUMMARY_ONLY}" -eq 1 ]; then
  echo "[DukeS3PSF-TestStage3Local] OUTPUT_BASE=${OUTPUT_BASE}"
  echo "[DukeS3PSF-TestStage3Local] EVAL_BASE=${EVAL_BASE}"
  summarize_results
  exit 0
fi

echo "[DukeS3PSF-TestStage3Local] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[DukeS3PSF-TestStage3Local] EVAL_BASE=${EVAL_BASE}"
echo "[DukeS3PSF-TestStage3Local] Summary files: ${SUMMARY_TSV}, ${SUMMARY_CSV}"
echo "[DukeS3PSF-TestStage3Local] WEIGHT_EPOCH=${WEIGHT_EPOCH} TEST_BATCH=${TEST_BATCH} INFER_WEIGHT=${INFER_WEIGHT} SKIP_COMPLETED=${SKIP_COMPLETED}"

running=0
failures=0
for idx in "${!EXPERIMENTS[@]}"; do
  run_experiment "${idx}" "${EXPERIMENTS[$idx]}" &
  running=$((running + 1))

  if [ "${running}" -ge "${MAX_JOBS}" ]; then
    if ! wait -n; then
      failures=1
    fi
    running=$((running - 1))
  fi
done

while [ "${running}" -gt 0 ]; do
  if ! wait -n; then
    failures=1
  fi
  running=$((running - 1))
done

summarize_results
echo "[DukeS3PSF-TestStage3Local] All tests finished."

if [ "${failures}" -ne 0 ]; then
  echo "[DukeS3PSF-TestStage3Local] One or more tests failed. Check logs under ${EVAL_BASE}."
  exit 1
fi
