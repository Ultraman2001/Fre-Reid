#!/usr/bin/env bash
# Train DukeMTMC-reID gated Stage-FCU checks on the current best map-Mamba/MSEF fusion.
# Usage:
#   bash scripts/run_duke_osnet_fdmf_msef_gated_stage_fcu.sh 0 1
#   bash scripts/run_duke_osnet_fdmf_msef_gated_stage_fcu.sh 0,1 2 ./logs/Duke/my_output
#   bash scripts/run_duke_osnet_fdmf_msef_gated_stage_fcu.sh summary ./logs/Duke/my_output
# The script writes summary.tsv and summary.csv under OUTPUT_BASE by default.
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
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_ARG:-./logs/Duke/osnet_fdmf_msef_gated_stage_fcu}}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
COMPLETE_EPOCH="${COMPLETE_EPOCH:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SUMMARY_TSV="${OUTPUT_BASE}/summary.tsv"
SUMMARY_CSV="${OUTPUT_BASE}/summary.csv"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

# name|gate_type|gate_reduction|gate_init_bias|init_scale|stages|direction|osnet_weight|fused_weight
declare -a EXPERIMENTS=(
  "plain_s2o2m_s3m2o|none|16|0.0|0.1|[2,3]|s2_o2m_s3_m2o|0.5|1.0"
  "gated_ch_b0_a010|channel|16|0.0|0.1|[2,3]|s2_o2m_s3_m2o|0.5|1.0"
  "gated_ch_b0_a020|channel|16|0.0|0.2|[2,3]|s2_o2m_s3_m2o|0.5|1.0"
  "gated_ch_bneg1_a010|channel|16|-1.0|0.1|[2,3]|s2_o2m_s3_m2o|0.5|1.0"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name gate_type gate_reduction gate_init_bias init_scale stages direction osnet_weight fused_weight

  IFS='|' read -r name gate_type gate_reduction gate_init_bias init_scale stages direction osnet_weight fused_weight <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"
  local train_log="${output_dir}/train_log.txt"
  local fcu_direction="${direction}"
  local stage2_direction=""
  local stage3_direction=""
  local osnet_pretrain_opts=()

  if [ "${direction}" = "s2_o2m_s3_m2o" ]; then
    fcu_direction="bidirectional"
    stage2_direction="osnet_to_mamba"
    stage3_direction="mamba_to_osnet"
  else
    stage2_direction="${fcu_direction}"
    stage3_direction="${fcu_direction}"
  fi

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${train_log}" ] && grep -q "Validation (Regular) Results - Epoch:[[:space:]]*${COMPLETE_EPOCH}" "${train_log}"; then
    echo "[DukeGatedStageFCU] SKIP completed EXP=${name} output=${output_dir} epoch=${COMPLETE_EPOCH}"
    return 0
  fi

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi

  echo "[DukeGatedStageFCU] GPU=${gpu} EXP=${name} output=${output_dir} gate=${gate_type} red=${gate_reduction} bias=${gate_init_bias} stages=${stages} stage2=${stage2_direction} stage3=${stage3_direction} init=${init_scale}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
    MODEL.OSNET_FUSION.FUSION_TYPE "'fdmf'" \
    MODEL.OSNET_FUSION.FUSION_NORM "'none'" \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT "${osnet_weight}" \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT "${fused_weight}" \
    MODEL.OSNET_FUSION.FCU_ENABLED True \
    MODEL.OSNET_FUSION.FCU_STAGES "${stages}" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'${fcu_direction}'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'${stage2_direction}'" \
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'${stage3_direction}'" \
    MODEL.OSNET_FUSION.FCU_INIT_SCALE "${init_scale}" \
    MODEL.OSNET_FUSION.FCU_GATE_TYPE "'${gate_type}'" \
    MODEL.OSNET_FUSION.FCU_GATE_REDUCTION "${gate_reduction}" \
    MODEL.OSNET_FUSION.FCU_GATE_INIT_BIAS "${gate_init_bias}" \
    MODEL.OSNET_FUSION.FDMF_FILTER_TYPE "'none'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
    TEST.FEAT_MODE "'mamba_fdmf_osnet'" \
    TEST.EVAL_ALL_FEATS False \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    "${osnet_pretrain_opts[@]}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local specs
  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[DukeGatedStageFCU] Summarizing results -> ${SUMMARY_TSV} and ${SUMMARY_CSV}"

  DUKE_GATED_STAGE_FCU_SPECS="${specs}" python - "${OUTPUT_BASE}" "${SUMMARY_TSV}" "${SUMMARY_CSV}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("DUKE_GATED_STAGE_FCU_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "gate_type", "gate_reduction", "gate_init_bias",
    "init_scale", "stages", "direction", "osnet_weight", "fused_weight",
    "last_epoch", "last_mAP", "last_R1", "last_R5", "last_R10",
    "best_epoch", "best_mAP", "best_R1", "best_R5", "best_R10",
    "log_file",
]

epoch_re = re.compile(r"Validation \(Regular\) Results - Epoch:\s*(\d+)")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank_re = re.compile(r"Rank-(1|5|10)\s*:?\s*([0-9.]+)%")
block_re = re.compile(r"(Validation .*Results|=== .*Results ===|MAMBA_FDMF_OSNET Results)")

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

def empty_metrics(status, log_file):
    return {
        "status": status,
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
        "log_file": log_file,
    }

rows = []
for spec in specs:
    (
        name,
        gate_type,
        gate_reduction,
        gate_init_bias,
        init_scale,
        stages,
        direction,
        osnet_weight,
        fused_weight,
    ) = spec.split("|")
    output_dir = os.path.join(output_base, name)
    train_log = os.path.join(output_dir, "train_log.txt")
    test_log = os.path.join(output_dir, "test_log.txt")
    log_file = train_log if os.path.exists(train_log) else test_log

    base = {
        "name": name,
        "gate_type": gate_type,
        "gate_reduction": gate_reduction,
        "gate_init_bias": gate_init_bias,
        "init_scale": init_scale,
        "stages": stages,
        "direction": direction,
        "osnet_weight": osnet_weight,
        "fused_weight": fused_weight,
    }

    if not os.path.exists(log_file):
        rows.append({**base, **empty_metrics("missing_log", "NA")})
        continue

    records = parse_records(log_file)
    if not records:
        rows.append({**base, **empty_metrics("no_eval_found", log_file)})
        continue

    last = records[-1]
    best = max(records, key=lambda item: float(item["mAP"]))
    rows.append({
        **base,
        "status": "ok",
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
        "log_file": log_file,
    })

for path, dialect in ((summary_tsv, "excel-tab"), (summary_csv, "excel")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect=dialect)
        writer.writeheader()
        writer.writerows(rows)

print()
print(
    f"{'name':<24} {'status':<12} {'gate':<8} {'bias':<7} {'init':<6} "
    f"{'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} "
    f"{'best_ep':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<24} {row['status']:<12} {row['gate_type']:<8} "
        f"{row['gate_init_bias']:<7} {row['init_scale']:<6} {row['last_epoch']:<8} "
        f"{row['last_mAP']:<8} {row['last_R1']:<8} {row['best_epoch']:<8} "
        f"{row['best_mAP']:<8} {row['best_R1']:<8}"
    )
PY
}

if [ "${SUMMARY_ONLY}" -eq 1 ]; then
  echo "[DukeGatedStageFCU] OUTPUT_BASE=${OUTPUT_BASE}"
  summarize_results
  exit 0
fi

echo "[DukeGatedStageFCU] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[DukeGatedStageFCU] Summary files: ${SUMMARY_TSV}, ${SUMMARY_CSV}"
echo "[DukeGatedStageFCU] SKIP_COMPLETED=${SKIP_COMPLETED} COMPLETE_EPOCH=${COMPLETE_EPOCH} TEST_BATCH=${TEST_BATCH}"

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
echo "[DukeGatedStageFCU] All experiments finished."

if [ "${failures}" -ne 0 ]; then
  echo "[DukeGatedStageFCU] One or more experiments failed. Check the logs above and ${OUTPUT_BASE}."
  exit 1
fi
