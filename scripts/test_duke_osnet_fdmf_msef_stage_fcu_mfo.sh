#!/usr/bin/env bash
# Test trained DukeMTMC map-Mamba/MSEF + Stage-FCU runs with
# inference-only MambaVision + FDMF-map + OSNet descriptor.
# Usage:
#   bash scripts/test_duke_osnet_fdmf_msef_stage_fcu_mfo.sh 0,1 2
#   bash scripts/test_duke_osnet_fdmf_msef_stage_fcu_mfo.sh 0,1 2 ./logs/Duke/osnet_fdmf_msef_stage_fcu_directional
#   bash scripts/test_duke_osnet_fdmf_msef_stage_fcu_mfo.sh summary ./logs/Duke/osnet_fdmf_msef_stage_fcu_directional
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
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_ARG:-./logs/Duke/osnet_fdmf_msef_stage_fcu_directional}}"
EVAL_BASE="${EVAL_BASE:-${OUTPUT_BASE}/eval_mamba_fdmf_osnet}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
WEIGHT_EPOCH="${WEIGHT_EPOCH:-160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SUMMARY_TSV="${EVAL_BASE}/summary.tsv"
SUMMARY_CSV="${EVAL_BASE}/summary.csv"

mkdir -p "${EVAL_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "fdmf_msef_s2_o2m|[2]|osnet_to_mamba|0.1|0.5|1.0"
  "fdmf_msef_s2_m2o|[2]|mamba_to_osnet|0.1|0.5|1.0"
  "fdmf_msef_s2_bidir|[2]|bidirectional|0.1|0.5|1.0"
  "fdmf_msef_s3_o2m|[3]|osnet_to_mamba|0.1|0.5|1.0"
  "fdmf_msef_s3_m2o|[3]|mamba_to_osnet|0.1|0.5|1.0"
  "fdmf_msef_s3_bidir|[3]|bidirectional|0.1|0.5|1.0"
  "fdmf_msef_s23_bidir|[2,3]|bidirectional|0.1|0.5|1.0"
  "fdmf_msef_s2o2m_s3m2o|[2,3]|s2_o2m_s3_m2o|0.1|0.5|1.0"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name stages direction init_scale osnet_weight fused_weight

  IFS='|' read -r name stages direction init_scale osnet_weight fused_weight <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local train_dir="${OUTPUT_BASE}/${name}"
  local weight_path="${train_dir}/transformer_${WEIGHT_EPOCH}.pth"
  local eval_dir="${EVAL_BASE}/${name}"
  local test_log="${eval_dir}/test_log.txt"
  local fcu_direction="${direction}"
  local stage2_direction=""
  local stage3_direction=""
  local osnet_pretrain_opts=()
  local fcu_stage_direction_opts=()

  if [ "${direction}" = "s2_o2m_s3_m2o" ]; then
    fcu_direction="bidirectional"
    stage2_direction="osnet_to_mamba"
    stage3_direction="mamba_to_osnet"
  else
    stage2_direction="${fcu_direction}"
    stage3_direction="${fcu_direction}"
  fi
  fcu_stage_direction_opts=(
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'${stage2_direction}'"
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'${stage3_direction}'"
  )

  if [ ! -f "${weight_path}" ]; then
    echo "[DukeFDMFMSEFStageFCU-TestMFO] Missing weight EXP=${name} weight=${weight_path}"
    return 0
  fi

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${test_log}" ] && grep -q "MAMBA_FDMF_OSNET Results" "${test_log}"; then
    echo "[DukeFDMFMSEFStageFCU-TestMFO] SKIP completed EXP=${name} log=${test_log}"
    return 0
  fi

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi

  echo "[DukeFDMFMSEFStageFCU-TestMFO] GPU=${gpu} EXP=${name} weight=${weight_path} eval=${eval_dir} stages=${stages} direction=${direction} stage2_dir=${stage2_direction} stage3_dir=${stage3_direction} batch=${TEST_BATCH}"

  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
    MODEL.OSNET_FUSION.FUSION_TYPE "'fdmf'" \
    MODEL.OSNET_FUSION.FUSION_NORM "'none'" \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT "${osnet_weight}" \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT "${fused_weight}" \
    MODEL.OSNET_FUSION.FCU_ENABLED True \
    MODEL.OSNET_FUSION.FCU_STAGES "${stages}" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'${fcu_direction}'" \
    "${fcu_stage_direction_opts[@]}" \
    MODEL.OSNET_FUSION.FCU_INIT_SCALE "${init_scale}" \
    MODEL.OSNET_FUSION.FDMF_FILTER_TYPE "'none'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE 0.1 \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL True \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO 16 \
    MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED False \
    TEST.WEIGHT "'${weight_path}'" \
    TEST.FEAT_MODE "'mamba_fdmf_osnet'" \
    TEST.EVAL_ALL_FEATS False \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    "${osnet_pretrain_opts[@]}" \
    OUTPUT_DIR "${eval_dir}"
}

summarize_results() {
  local specs
  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[DukeFDMFMSEFStageFCU-TestMFO] Summarizing results -> ${SUMMARY_TSV} and ${SUMMARY_CSV}"

  DUKE_FDMF_MSEF_STAGE_FCU_SPECS="${specs}" python - "${OUTPUT_BASE}" "${EVAL_BASE}" "${SUMMARY_TSV}" "${SUMMARY_CSV}" "${WEIGHT_EPOCH}" <<'PY'
import csv
import os
import re
import sys

output_base, eval_base, summary_tsv, summary_csv, weight_epoch = sys.argv[1:6]
specs = [line for line in os.environ.get("DUKE_FDMF_MSEF_STAGE_FCU_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "stages", "direction", "init_scale", "osnet_weight", "fused_weight",
    "weight_file", "test_log",
    "last_epoch", "last_mAP", "last_R1", "last_R5", "last_R10",
    "best_epoch", "best_mAP", "best_R1", "best_R5", "best_R10",
]

epoch_re = re.compile(r"Validation \(Regular\) Results - Epoch:\s*(\d+)")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank_re = re.compile(r"Rank-(1|5|10)\s*:?\s*([0-9.]+)%")
block_re = re.compile(r"(MAMBA_FDMF_OSNET Results|Validation .*Results|=== .*Results ===)")

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
    name, stages, direction, init_scale, osnet_weight, fused_weight = spec.split("|")
    weight_file = os.path.join(output_base, name, f"transformer_{weight_epoch}.pth")
    test_log = os.path.join(eval_base, name, "test_log.txt")
    base = {
        "name": name,
        "stages": stages,
        "direction": direction,
        "init_scale": init_scale,
        "osnet_weight": osnet_weight,
        "fused_weight": fused_weight,
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
    f"{'name':<26} {'status':<14} {'stages':<7} {'dir':<15} "
    f"{'last_mAP':<8} {'last_R1':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<26} {row['status']:<14} {row['stages']:<7} {row['direction']:<15} "
        f"{row['last_mAP']:<8} {row['last_R1']:<8} {row['best_mAP']:<8} {row['best_R1']:<8}"
    )
PY
}

if [ "${SUMMARY_ONLY}" -eq 1 ]; then
  echo "[DukeFDMFMSEFStageFCU-TestMFO] OUTPUT_BASE=${OUTPUT_BASE}"
  echo "[DukeFDMFMSEFStageFCU-TestMFO] EVAL_BASE=${EVAL_BASE}"
  summarize_results
  exit 0
fi

echo "[DukeFDMFMSEFStageFCU-TestMFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[DukeFDMFMSEFStageFCU-TestMFO] EVAL_BASE=${EVAL_BASE}"
echo "[DukeFDMFMSEFStageFCU-TestMFO] Summary files: ${SUMMARY_TSV}, ${SUMMARY_CSV}"
echo "[DukeFDMFMSEFStageFCU-TestMFO] WEIGHT_EPOCH=${WEIGHT_EPOCH} TEST_BATCH=${TEST_BATCH} SKIP_COMPLETED=${SKIP_COMPLETED}"

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
echo "[DukeFDMFMSEFStageFCU-TestMFO] All tests finished."

if [ "${failures}" -ne 0 ]; then
  echo "[DukeFDMFMSEFStageFCU-TestMFO] One or more tests failed. Check logs under ${EVAL_BASE}."
  exit 1
fi
