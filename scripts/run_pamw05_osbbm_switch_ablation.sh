#!/usr/bin/env bash
# Run PADE-style PAM(w=0.5) + OSBBM-m2 switch-length ablations.
set -euo pipefail

if [ "${1:-}" = "--summary-only" ] || [ "${1:-}" = "summary" ]; then
  SUMMARY_ONLY=1
  GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
  MAX_JOBS=2
else
  SUMMARY_ONLY=0
  GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
  MAX_JOBS="${2:-2}"
fi

CONFIG="${CONFIG:-configs/OCC_Duke/mambavision_tiny_transreid_pam_padeaug_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/pamw05_osbbm_switch_ablation}"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "pamw05_m2_c21_120_p20_on10|0.25|8|2|0.50|base|cycle|21|120|20|10"
  "pamw05_m2_c21_120_p20_on5|0.25|8|2|0.50|base|cycle|21|120|20|5"
  "pamw05_m2_c21_120_p20_on15|0.25|8|2|0.50|base|cycle|21|120|20|15"
  "pamw05_m2_c21_120_p10_on5|0.25|8|2|0.50|base|cycle|21|120|10|5"
  "pamw05_m2_c21_120_p30_on10|0.25|8|2|0.50|base|cycle|21|120|30|10"
  "pamw05_m2_c21_120_p40_on20|0.25|8|2|0.50|base|cycle|21|120|40|20"
  "pamw05_m2_c41_120_p20_on10|0.25|8|2|0.50|base|cycle|41|120|20|10"
  "pamw05_m2_c21_140_p20_on10|0.25|8|2|0.50|base|cycle|21|140|20|10"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name prob blocks mix_blocks gray_prob apply_to schedule start_epoch end_epoch period_epochs on_epochs

  IFS='|' read -r name prob blocks mix_blocks gray_prob apply_to schedule start_epoch end_epoch period_epochs on_epochs <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"

  echo "[PAMw05-OSBBM-Switch] GPU=${gpu} EXP=${name} schedule=${schedule} start=${start_epoch} end=${end_epoch} period=${period_epochs} on=${on_epochs}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    INPUT.PAM.ENABLED True \
    INPUT.PAM.AUG_MODE "'pade'" \
    INPUT.OSBBM.ENABLED True \
    INPUT.OSBBM.PROB "${prob}" \
    INPUT.OSBBM.NUM_BLOCKS "${blocks}" \
    INPUT.OSBBM.NUM_MIX_BLOCKS "${mix_blocks}" \
    INPUT.OSBBM.GRAY_PROB "${gray_prob}" \
    INPUT.OSBBM.APPLY_TO "'${apply_to}'" \
    INPUT.OSBBM.SCHEDULE "'${schedule}'" \
    INPUT.OSBBM.START_EPOCH "${start_epoch}" \
    INPUT.OSBBM.END_EPOCH "${end_epoch}" \
    INPUT.OSBBM.PERIOD_EPOCHS "${period_epochs}" \
    INPUT.OSBBM.ON_EPOCHS "${on_epochs}" \
    SOLVER.PAM_AUGMENTED_LOSS_WEIGHT 0.5 \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local summary_file="${OUTPUT_BASE}/summary.tsv"
  local summary_csv="${OUTPUT_BASE}/summary.csv"
  local specs

  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[PAMw05-OSBBM-Switch] Summarizing results -> ${summary_file} and ${summary_csv}"

  PAMW05_OSBBM_SWITCH_SPECS="${specs}" python - "${OUTPUT_BASE}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("PAMW05_OSBBM_SWITCH_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "prob", "blocks", "mix_blocks", "gray_prob", "apply_to",
    "schedule", "start_epoch", "end_epoch", "period_epochs", "on_epochs",
    "off_epochs", "last_epoch", "last_mAP", "last_R1", "last_R5", "last_R10",
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
    (
        name, prob, blocks, mix_blocks, gray_prob, apply_to,
        schedule, start_epoch, end_epoch, period_epochs, on_epochs,
    ) = spec.split("|")
    output_dir = os.path.join(output_base, name)
    train_log = os.path.join(output_dir, "train_log.txt")
    test_log = os.path.join(output_dir, "test_log.txt")
    log_file = train_log if os.path.exists(train_log) else test_log

    off_epochs = str(max(int(period_epochs) - int(on_epochs), 0))
    base = {
        "name": name,
        "prob": prob,
        "blocks": blocks,
        "mix_blocks": mix_blocks,
        "gray_prob": gray_prob,
        "apply_to": apply_to,
        "schedule": schedule,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "period_epochs": period_epochs,
        "on_epochs": on_epochs,
        "off_epochs": off_epochs,
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
    f"{'name':<32} {'status':<12} {'start':<6} {'end':<5} "
    f"{'on':<4} {'off':<4} {'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} "
    f"{'best_ep':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<32} {row['status']:<12} {row['start_epoch']:<6} {row['end_epoch']:<5} "
        f"{row['on_epochs']:<4} {row['off_epochs']:<4} {row['last_epoch']:<8} {row['last_mAP']:<8} {row['last_R1']:<8} "
        f"{row['best_epoch']:<8} {row['best_mAP']:<8} {row['best_R1']:<8}"
    )
PY
}

if [ "${SUMMARY_ONLY}" -eq 1 ]; then
  summarize_results
  exit 0
fi

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
echo "[PAMw05-OSBBM-Switch] All experiments finished."

if [ "${failures}" -ne 0 ]; then
  echo "[PAMw05-OSBBM-Switch] One or more experiments failed. Check the logs above and ${OUTPUT_BASE}."
  exit 1
fi
