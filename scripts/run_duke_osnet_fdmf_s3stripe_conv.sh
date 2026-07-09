#!/usr/bin/env bash
# Ablate stripe-local convolution enhancement before Stage3 stripe Mamba.
# Usage:
#   bash scripts/run_duke_osnet_fdmf_s3stripe_conv.sh 0,1 2
#   bash scripts/run_duke_osnet_fdmf_s3stripe_conv.sh 0,1 2 ./logs/Duke/my_output
#   bash scripts/run_duke_osnet_fdmf_s3stripe_conv.sh summary ./logs/Duke/my_output
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
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_ARG:-./logs/Duke/osnet_fdmf_s3stripe_conv}}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
COMPLETE_EPOCH="${COMPLETE_EPOCH:-160}"
INFER_WEIGHT="${INFER_WEIGHT:-0.5}"
SUMMARY_TSV="${OUTPUT_BASE}/summary.tsv"
SUMMARY_CSV="${OUTPUT_BASE}/summary.csv"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

# name|conv_type|mamba_depth
declare -a EXPERIMENTS=(
  "s3psf_k2_w01_d1_noconv|none|1"
  "s3psf_k2_w01_d1_dw3|dw3|1"
  "s3psf_k2_w01_d1_incep|inception|1"
  "s3psf_k2_w01_d0_incep|inception|0"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name conv_type mamba_depth

  IFS='|' read -r name conv_type mamba_depth <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"
  local train_log="${output_dir}/train_log.txt"
  local osnet_pretrain_opts=()

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${train_log}" ] && grep -q "Validation (Regular) Results - Epoch:[[:space:]]*${COMPLETE_EPOCH}" "${train_log}"; then
    echo "[DukeS3PSFConv] SKIP completed EXP=${name} output=${output_dir} epoch=${COMPLETE_EPOCH}"
    return 0
  fi

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi

  echo "[DukeS3PSFConv] GPU=${gpu} EXP=${name} output=${output_dir} conv=${conv_type} depth=${mamba_depth} infer_w=${INFER_WEIGHT}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
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
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${INFER_WEIGHT}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH "${mamba_depth}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS True \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_SUPERVISION_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LG_ENHANCE_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONV_ENHANCE_TYPE "'${conv_type}'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONV_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_ENABLED False \
    TEST.FEAT_MODE "'mamba_fdmf_osnet_stage3local'" \
    TEST.EVAL_ALL_FEATS False \
    "${osnet_pretrain_opts[@]}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local specs
  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[DukeS3PSFConv] Summarizing results -> ${SUMMARY_TSV} and ${SUMMARY_CSV}"

  INFER_WEIGHT="${INFER_WEIGHT}" DUKE_S3PSF_CONV_SPECS="${specs}" python - "${OUTPUT_BASE}" "${SUMMARY_TSV}" "${SUMMARY_CSV}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("DUKE_S3PSF_CONV_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "k", "w", "dep", "conv", "infer_w",
    "last_ep", "last_mAP", "last_R1", "best_ep", "best_mAP", "best_R1",
]

epoch_re = re.compile(r"Validation \(Regular\) Results - Epoch:\s*(\d+)")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank1_re = re.compile(r"Rank-1\s*:?\s*([0-9.]+)%")

def parse_records(log_path):
    records = []
    current = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = epoch_re.search(line)
            if epoch_match:
                if current and "mAP" in current and "R1" in current:
                    records.append(current)
                current = {"epoch": epoch_match.group(1)}
                continue
            if current is None:
                continue
            map_match = map_re.search(line)
            if map_match:
                current["mAP"] = map_match.group(1)
                continue
            rank_match = rank1_re.search(line)
            if rank_match:
                current["R1"] = rank_match.group(1)
                if "mAP" in current:
                    records.append(current)
                    current = None
    if current and "mAP" in current and "R1" in current:
        records.append(current)
    return records

rows = []
for spec in specs:
    name, conv, dep = spec.split("|")
    log_path = os.path.join(output_base, name, "train_log.txt")
    row = {
        "name": name,
        "status": "missing",
        "k": "2",
        "w": "0.1",
        "dep": dep,
        "conv": conv,
        "infer_w": os.environ.get("INFER_WEIGHT", "0.5"),
        "last_ep": "NA",
        "last_mAP": "NA",
        "last_R1": "NA",
        "best_ep": "NA",
        "best_mAP": "NA",
        "best_R1": "NA",
    }
    if os.path.exists(log_path):
        records = parse_records(log_path)
        if records:
            row["status"] = "ok"
            last = records[-1]
            best = max(records, key=lambda r: float(r["mAP"]))
            row.update({
                "last_ep": last["epoch"],
                "last_mAP": last["mAP"],
                "last_R1": last["R1"],
                "best_ep": best["epoch"],
                "best_mAP": best["mAP"],
                "best_R1": best["R1"],
            })
        else:
            row["status"] = "no_eval"
    rows.append(row)

for path, delimiter in ((summary_tsv, "\t"), (summary_csv, ",")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)

widths = {field: max(len(field), *(len(str(row[field])) for row in rows)) for field in fields}
print(" ".join(field.ljust(widths[field]) for field in fields))
for row in rows:
    print(" ".join(str(row[field]).ljust(widths[field]) for field in fields))
PY
}

if [ "${SUMMARY_ONLY}" = "0" ]; then
  active_jobs=0
  for idx in "${!EXPERIMENTS[@]}"; do
    run_experiment "${idx}" "${EXPERIMENTS[$idx]}" &
    active_jobs=$((active_jobs + 1))
    if [ "${active_jobs}" -ge "${MAX_JOBS}" ]; then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
  wait
fi

summarize_results
