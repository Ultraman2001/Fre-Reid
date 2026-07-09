#!/usr/bin/env bash
# Re-evaluate the four Stage3 stripe-conv ablation checkpoints with multiple
# stripe-local inference weights.
# Usage:
#   bash scripts/test_duke_osnet_fdmf_s3stripe_conv_infer_sweep.sh 0,1 2
#   bash scripts/test_duke_osnet_fdmf_s3stripe_conv_infer_sweep.sh 0,1 2 ./logs/Duke/osnet_fdmf_s3stripe_conv
#   INFER_WEIGHTS="0.3 0.5 0.7" bash scripts/test_duke_osnet_fdmf_s3stripe_conv_infer_sweep.sh 0,1 2
#   bash scripts/test_duke_osnet_fdmf_s3stripe_conv_infer_sweep.sh summary ./logs/Duke/osnet_fdmf_s3stripe_conv
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
EVAL_BASE="${EVAL_BASE:-${OUTPUT_BASE}/eval_infer_sweep}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
WEIGHT_EPOCH="${WEIGHT_EPOCH:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
INFER_WEIGHTS="${INFER_WEIGHTS:-0.3 0.5 0.7}"
SUMMARY_TSV="${EVAL_BASE}/summary.tsv"
SUMMARY_CSV="${EVAL_BASE}/summary.csv"

mkdir -p "${EVAL_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

read -r -a INFER_WEIGHT_ARRAY <<< "${INFER_WEIGHTS}"
if [ "${#INFER_WEIGHT_ARRAY[@]}" -eq 0 ]; then
  INFER_WEIGHT_ARRAY=("0.3" "0.5" "0.7")
fi

# name|conv_type|mamba_depth
declare -a EXPERIMENTS=(
  "s3psf_k2_w01_d1_noconv|none|1"
  "s3psf_k2_w01_d1_dw3|dw3|1"
  "s3psf_k2_w01_d1_incep|inception|1"
  "s3psf_k2_w01_d0_incep|inception|0"
)

infer_tag() {
  python - "$1" <<'PY'
import sys

value = float(sys.argv[1])
print(f"w{int(round(value * 100)):03d}")
PY
}

run_experiment() {
  local idx="$1"
  local spec="$2"
  local infer_weight="$3"
  local name conv_type mamba_depth

  IFS='|' read -r name conv_type mamba_depth <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local train_dir="${OUTPUT_BASE}/${name}"
  local weight_path="${train_dir}/transformer_${WEIGHT_EPOCH}.pth"
  local tag
  tag="$(infer_tag "${infer_weight}")"
  local eval_dir="${EVAL_BASE}/${tag}/${name}"
  local test_log="${eval_dir}/test_log.txt"
  local osnet_pretrain_opts=()

  if [ ! -f "${weight_path}" ]; then
    echo "[DukeS3PSFConvInfer] Missing weight EXP=${name} weight=${weight_path}"
    return 0
  fi

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${test_log}" ] && grep -q "MAMBA_FDMF_OSNET_STAGE3LOCAL Results" "${test_log}"; then
    echo "[DukeS3PSFConvInfer] SKIP completed EXP=${name} infer_w=${infer_weight} log=${test_log}"
    return 0
  fi

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi

  echo "[DukeS3PSFConvInfer] GPU=${gpu} EXP=${name} weight=${weight_path} eval=${eval_dir} conv=${conv_type} depth=${mamba_depth} infer_w=${infer_weight}"

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
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${infer_weight}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH "${mamba_depth}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS True \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM 0 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_SUPERVISION_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LG_ENHANCE_ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONV_ENHANCE_TYPE "'${conv_type}'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONV_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_ENABLED False \
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
  echo "[DukeS3PSFConvInfer] Summarizing results -> ${SUMMARY_TSV} and ${SUMMARY_CSV}"

  DUKE_S3PSF_CONV_SPECS="${specs}" \
  DUKE_S3PSF_INFER_WEIGHTS="${INFER_WEIGHTS}" \
  python - "${OUTPUT_BASE}" "${EVAL_BASE}" "${SUMMARY_TSV}" "${SUMMARY_CSV}" "${WEIGHT_EPOCH}" <<'PY'
import csv
import os
import re
import sys

output_base, eval_base, summary_tsv, summary_csv, weight_epoch = sys.argv[1:6]
specs = [line for line in os.environ.get("DUKE_S3PSF_CONV_SPECS", "").splitlines() if line.strip()]
infer_weights = os.environ.get("DUKE_S3PSF_INFER_WEIGHTS", "0.3 0.5 0.7").split()

fields = [
    "name", "status", "k", "w", "dep", "conv", "infer_w",
    "weight_file", "test_log",
    "mAP", "R1", "R5", "R10",
]

block_re = re.compile(r"===\s+MAMBA_FDMF_OSNET_STAGE3LOCAL Results\s+===")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank_re = re.compile(r"Rank-(1|5|10)\s*:?\s*([0-9.]+)%")

def infer_tag(value):
    return "w{:03d}".format(int(round(float(value) * 100)))

def parse_eval(log_path):
    in_block = False
    record = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if block_re.search(line):
                in_block = True
                record = {}
                continue
            if not in_block:
                continue
            map_match = map_re.search(line)
            if map_match:
                record["mAP"] = map_match.group(1)
                continue
            rank_match = rank_re.search(line)
            if rank_match:
                rank, value = rank_match.groups()
                record[f"R{rank}"] = value
                if all(key in record for key in ("mAP", "R1", "R5", "R10")):
                    return record
    return record if all(key in record for key in ("mAP", "R1", "R5", "R10")) else None

def empty_metrics(status, weight_file, test_log):
    return {
        "status": status,
        "weight_file": weight_file,
        "test_log": test_log,
        "mAP": "NA",
        "R1": "NA",
        "R5": "NA",
        "R10": "NA",
    }

rows = []
for spec in specs:
    name, conv, dep = spec.split("|")
    weight_file = os.path.join(output_base, name, f"transformer_{weight_epoch}.pth")
    for infer_w in infer_weights:
        tag = infer_tag(infer_w)
        test_log = os.path.join(eval_base, tag, name, "test_log.txt")
        base = {
            "name": name,
            "k": "2",
            "w": "0.1",
            "dep": dep,
            "conv": conv,
            "infer_w": infer_w,
        }
        if not os.path.exists(weight_file):
            rows.append({**base, **empty_metrics("missing_weight", weight_file, test_log)})
            continue
        if not os.path.exists(test_log):
            rows.append({**base, **empty_metrics("missing_log", weight_file, test_log)})
            continue
        record = parse_eval(test_log)
        if record is None:
            rows.append({**base, **empty_metrics("no_eval_found", weight_file, test_log)})
            continue
        rows.append({
            **base,
            "status": "ok",
            "weight_file": weight_file,
            "test_log": test_log,
            "mAP": record["mAP"],
            "R1": record["R1"],
            "R5": record["R5"],
            "R10": record["R10"],
        })

for path, delimiter in ((summary_tsv, "\t"), (summary_csv, ",")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)

widths = {field: max(len(field), *(len(str(row[field])) for row in rows)) for field in fields}
print(" ".join(field.ljust(widths[field]) for field in fields))
for row in rows:
    print(" ".join(str(row[field]).ljust(widths[field]) for field in fields))

valid = [row for row in rows if row["status"] == "ok"]
if valid:
    best_map = max(valid, key=lambda row: float(row["mAP"]))
    best_r1 = max(valid, key=lambda row: float(row["R1"]))
    print("")
    print("best_mAP: {name} infer_w={infer_w} conv={conv} dep={dep} mAP={mAP} R1={R1}".format(**best_map))
    print("best_R1 : {name} infer_w={infer_w} conv={conv} dep={dep} mAP={mAP} R1={R1}".format(**best_r1))
PY
}

if [ "${SUMMARY_ONLY}" = "0" ]; then
  active_jobs=0
  job_idx=0
  for infer_weight in "${INFER_WEIGHT_ARRAY[@]}"; do
    for spec in "${EXPERIMENTS[@]}"; do
      run_experiment "${job_idx}" "${spec}" "${infer_weight}" &
      active_jobs=$((active_jobs + 1))
      job_idx=$((job_idx + 1))
      if [ "${active_jobs}" -ge "${MAX_JOBS}" ]; then
        wait -n
        active_jobs=$((active_jobs - 1))
      fi
    done
  done
  wait
fi

summarize_results
