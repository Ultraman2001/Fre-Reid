#!/usr/bin/env bash
# Train DukeMTMC-reID MambaVision + OSNet frequency-decoupled Mamba fusion.
# Usage:
#   bash scripts/run_duke_osnet_fdmf.sh 0,1 2
#   bash scripts/run_duke_osnet_fdmf.sh 0,1 2 ./logs/Duke/my_output
#   bash scripts/run_duke_osnet_fdmf.sh summary ./logs/Duke/my_output
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
  MAX_JOBS="${2:-2}"
  OUTPUT_BASE_ARG="${3:-}"
fi

CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_ARG:-./logs/Duke/osnet_fdmf}}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
COMPLETE_EPOCH="${COMPLETE_EPOCH:-160}"
SUMMARY_TSV="${OUTPUT_BASE}/summary.tsv"
SUMMARY_CSV="${OUTPUT_BASE}/summary.csv"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "fdmf_raw_osw05_fuw1|0.5|1.0|64|5|3|dynamic|raw_fdmf|1|0.1|True"
  "fdmf_raw_osw1_fuw1|1.0|1.0|64|5|3|dynamic|raw_fdmf|1|0.1|True"
  "fdmf_raw_no_mamba|0.5|1.0|64|5|3|dynamic|raw_fdmf|0|0.1|True"
  "fdmf_raw_fixed_filter|0.5|1.0|64|5|3|fixed|raw_fdmf|1|0.1|True"
  "fdmf_only|0.5|1.0|64|5|3|dynamic|fdmf_only|1|0.1|True"
  "fdmf_mamba_fdmf|0.5|1.0|64|5|3|dynamic|mamba_fdmf|1|0.1|True"
  "fdmf_mamba_no_mamba|0.5|1.0|64|5|3|dynamic|mamba_fdmf|0|0.1|True"
  "fdmf_msef_stage_fcu|0.5|1.0|64|5|3|none|mamba_fdmf|1|0.1|True|True|[2,3]|bidirectional|osnet_to_mamba|mamba_to_osnet|0.1"
  "fdmf_raw_unidir|0.5|1.0|64|5|3|dynamic|raw_fdmf|1|0.1|False"
  "fdmf_raw_depth2|0.5|1.0|64|5|3|dynamic|raw_fdmf|2|0.1|True"
  "fdmf_raw_c32|0.5|1.0|32|5|3|dynamic|raw_fdmf|1|0.1|True"
  "fdmf_raw_c128|0.5|1.0|128|5|3|dynamic|raw_fdmf|1|0.1|True"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name osnet_weight fused_weight compressed low_kernel high_kernel filter_type fused_form mamba_depth init_scale bidirectional
  local fcu_enabled fcu_stages fcu_direction fcu_stage2_direction fcu_stage3_direction fcu_init_scale

  IFS='|' read -r name osnet_weight fused_weight compressed low_kernel high_kernel filter_type fused_form mamba_depth init_scale bidirectional fcu_enabled fcu_stages fcu_direction fcu_stage2_direction fcu_stage3_direction fcu_init_scale <<< "${spec}"
  fcu_enabled="${fcu_enabled:-False}"
  fcu_stages="${fcu_stages:-[2,3]}"
  fcu_direction="${fcu_direction:-bidirectional}"
  fcu_stage2_direction="${fcu_stage2_direction:-}"
  fcu_stage3_direction="${fcu_stage3_direction:-}"
  fcu_init_scale="${fcu_init_scale:-0.1}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"
  local train_log="${output_dir}/train_log.txt"
  local osnet_pretrain_opts=()
  local fcu_opts=(
    MODEL.OSNET_FUSION.FCU_ENABLED "${fcu_enabled}"
    MODEL.OSNET_FUSION.FCU_STAGES "${fcu_stages}"
    MODEL.OSNET_FUSION.FCU_DIRECTION "'${fcu_direction}'"
    MODEL.OSNET_FUSION.FCU_INIT_SCALE "${fcu_init_scale}"
  )

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${train_log}" ] && grep -q "Validation (Regular) Results - Epoch:[[:space:]]*${COMPLETE_EPOCH}" "${train_log}"; then
    echo "[DukeOSNetFDMF] SKIP completed EXP=${name} output=${output_dir} epoch=${COMPLETE_EPOCH}"
    return 0
  fi

  if [ -n "${OSNET_PRETRAIN}" ]; then
    osnet_pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi
  if [ -n "${fcu_stage2_direction}" ]; then
    fcu_opts+=(MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'${fcu_stage2_direction}'")
  fi
  if [ -n "${fcu_stage3_direction}" ]; then
    fcu_opts+=(MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'${fcu_stage3_direction}'")
  fi

  echo "[DukeOSNetFDMF] GPU=${gpu} EXP=${name} output=${output_dir} osnet_w=${osnet_weight} fused_w=${fused_weight} comp=${compressed} low_k=${low_kernel} high_k=${high_kernel} filter=${filter_type} form=${fused_form} depth=${mamba_depth} init=${init_scale} bidir=${bidirectional} fcu=${fcu_enabled} stages=${fcu_stages} fcu_dir=${fcu_direction} s2=${fcu_stage2_direction:-${fcu_direction}} s3=${fcu_stage3_direction:-${fcu_direction}}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
    MODEL.OSNET_FUSION.FUSION_TYPE "'fdmf'" \
    MODEL.OSNET_FUSION.FUSION_NORM "'none'" \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT "${osnet_weight}" \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT "${fused_weight}" \
    MODEL.OSNET_FUSION.FDMF_COMPRESSED_CHANNELS "${compressed}" \
    MODEL.OSNET_FUSION.FDMF_LOWPASS_KERNEL "${low_kernel}" \
    MODEL.OSNET_FUSION.FDMF_HIGHPASS_KERNEL "${high_kernel}" \
    MODEL.OSNET_FUSION.FDMF_FILTER_TYPE "'${filter_type}'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'${fused_form}'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH "${mamba_depth}" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE "${init_scale}" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL "${bidirectional}" \
    "${fcu_opts[@]}" \
    TEST.FEAT_MODE "'fdmf'" \
    "${osnet_pretrain_opts[@]}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local summary_file="${SUMMARY_TSV}"
  local summary_csv="${SUMMARY_CSV}"
  local specs

  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[DukeOSNetFDMF] Summarizing results -> ${summary_file} and ${summary_csv}"

  DUKE_OSNET_FDMF_SPECS="${specs}" python - "${OUTPUT_BASE}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("DUKE_OSNET_FDMF_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "osnet_weight", "fused_weight",
    "compressed", "low_kernel", "high_kernel", "filter_type", "fused_form",
    "mamba_depth", "init_scale", "bidirectional",
    "fcu_enabled", "fcu_stages", "fcu_direction", "fcu_stage2_direction", "fcu_stage3_direction", "fcu_init_scale",
    "last_epoch", "last_mAP", "last_R1", "last_R5", "last_R10",
    "best_epoch", "best_mAP", "best_R1", "best_R5", "best_R10",
    "log_file",
]

epoch_re = re.compile(r"Validation \(Regular\) Results - Epoch:\s*(\d+)")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank_re = re.compile(r"Rank-(1|5|10)\s*:?\s*([0-9.]+)%")
block_re = re.compile(r"(Validation .*Results|=== .*Results ===)")

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
    parts = spec.split("|")
    if len(parts) < 11:
        raise ValueError(f"Invalid FDMF spec: {spec}")
    (
        name,
        osnet_weight,
        fused_weight,
        compressed,
        low_kernel,
        high_kernel,
        filter_type,
        fused_form,
        mamba_depth,
        init_scale,
        bidirectional,
    ) = parts[:11]
    (
        fcu_enabled,
        fcu_stages,
        fcu_direction,
        fcu_stage2_direction,
        fcu_stage3_direction,
        fcu_init_scale,
    ) = (parts[11:] + ["False", "[2,3]", "bidirectional", "", "", "0.1"])[:6]
    output_dir = os.path.join(output_base, name)
    train_log = os.path.join(output_dir, "train_log.txt")
    test_log = os.path.join(output_dir, "test_log.txt")
    log_file = train_log if os.path.exists(train_log) else test_log

    base = {
        "name": name,
        "osnet_weight": osnet_weight,
        "fused_weight": fused_weight,
        "compressed": compressed,
        "low_kernel": low_kernel,
        "high_kernel": high_kernel,
        "filter_type": filter_type,
        "fused_form": fused_form,
        "mamba_depth": mamba_depth,
        "init_scale": init_scale,
        "bidirectional": bidirectional,
        "fcu_enabled": fcu_enabled,
        "fcu_stages": fcu_stages,
        "fcu_direction": fcu_direction,
        "fcu_stage2_direction": fcu_stage2_direction,
        "fcu_stage3_direction": fcu_stage3_direction,
        "fcu_init_scale": fcu_init_scale,
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
    f"{'name':<24} {'status':<12} {'osw':<5} {'fuw':<5} {'filter':<8} {'form':<9} "
    f"{'dep':<4} {'fcu':<5} {'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} "
    f"{'best_ep':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<24} {row['status']:<12} {row['osnet_weight']:<5} "
        f"{row['fused_weight']:<5} {row['filter_type']:<8} {row['fused_form']:<9} "
        f"{row['mamba_depth']:<4} {row['fcu_enabled']:<5} {row['last_epoch']:<8} "
        f"{row['last_mAP']:<8} {row['last_R1']:<8} {row['best_epoch']:<8} "
        f"{row['best_mAP']:<8} {row['best_R1']:<8}"
    )
PY
}

if [ "${SUMMARY_ONLY}" -eq 1 ]; then
  echo "[DukeOSNetFDMF] OUTPUT_BASE=${OUTPUT_BASE}"
  summarize_results
  exit 0
fi

echo "[DukeOSNetFDMF] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[DukeOSNetFDMF] Summary files: ${SUMMARY_TSV}, ${SUMMARY_CSV}"
echo "[DukeOSNetFDMF] SKIP_COMPLETED=${SKIP_COMPLETED} COMPLETE_EPOCH=${COMPLETE_EPOCH}"

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
echo "[DukeOSNetFDMF] All experiments finished."

if [ "${failures}" -ne 0 ]; then
  echo "[DukeOSNetFDMF] One or more experiments failed. Check the logs above and ${OUTPUT_BASE}."
  exit 1
fi
