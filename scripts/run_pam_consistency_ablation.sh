#!/usr/bin/env bash
# Run PAM branch feature-consistency ablations on top of PADE-style PAM(w=0.5).
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
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/pam_consistency_ablation}"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "baseline_no_osbbm|False|False|0.00|pairwise|True"
  "baseline_osbbm|True|False|0.00|pairwise|True"
  "pamc_w002_osbbm|True|True|0.02|pairwise|True"
  "pamc_w005_osbbm|True|True|0.05|pairwise|True"
  "pamc_w010_osbbm|True|True|0.10|pairwise|True"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name osbbm_enabled pamc_enabled pamc_weight pamc_mode detach_base

  IFS='|' read -r name osbbm_enabled pamc_enabled pamc_weight pamc_mode detach_base <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"

  echo "[PAM-Consistency] GPU=${gpu} EXP=${name} osbbm=${osbbm_enabled} pamc=${pamc_enabled} weight=${pamc_weight} mode=${pamc_mode}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    INPUT.PAM.ENABLED True \
    INPUT.PAM.AUG_MODE "'pade'" \
    INPUT.OSBBM.ENABLED "${osbbm_enabled}" \
    INPUT.OSBBM.PROB 0.25 \
    INPUT.OSBBM.NUM_BLOCKS 8 \
    INPUT.OSBBM.NUM_MIX_BLOCKS 2 \
    INPUT.OSBBM.GRAY_PROB 0.50 \
    INPUT.OSBBM.APPLY_TO "'base'" \
    INPUT.OSBBM.SCHEDULE "'cycle'" \
    INPUT.OSBBM.START_EPOCH 21 \
    INPUT.OSBBM.END_EPOCH 120 \
    INPUT.OSBBM.PERIOD_EPOCHS 20 \
    INPUT.OSBBM.ON_EPOCHS 10 \
    SOLVER.PAM_AUGMENTED_LOSS_WEIGHT 0.5 \
    SOLVER.PAM_CONSISTENCY_ENABLED "${pamc_enabled}" \
    SOLVER.PAM_CONSISTENCY_WEIGHT "${pamc_weight}" \
    SOLVER.PAM_CONSISTENCY_MODE "'${pamc_mode}'" \
    SOLVER.PAM_CONSISTENCY_DETACH_BASE "${detach_base}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local summary_file="${OUTPUT_BASE}/summary.tsv"
  local summary_csv="${OUTPUT_BASE}/summary.csv"
  local specs

  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[PAM-Consistency] Summarizing results -> ${summary_file} and ${summary_csv}"

  PAM_CONSISTENCY_SPECS="${specs}" python - "${OUTPUT_BASE}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("PAM_CONSISTENCY_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "osbbm_enabled", "pamc_enabled", "pamc_weight",
    "pamc_mode", "detach_base", "last_epoch", "last_mAP", "last_R1",
    "last_R5", "last_R10", "best_epoch", "best_mAP", "best_R1",
    "best_R5", "best_R10", "final_minus_best_mAP", "log_file",
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
        "final_minus_best_mAP": "NA",
        "log_file": log_file,
    }


rows = []
for spec in specs:
    name, osbbm_enabled, pamc_enabled, pamc_weight, pamc_mode, detach_base = spec.split("|")
    output_dir = os.path.join(output_base, name)
    train_log = os.path.join(output_dir, "train_log.txt")
    test_log = os.path.join(output_dir, "test_log.txt")
    log_file = train_log if os.path.exists(train_log) else test_log

    base = {
        "name": name,
        "osbbm_enabled": osbbm_enabled,
        "pamc_enabled": pamc_enabled,
        "pamc_weight": pamc_weight,
        "pamc_mode": pamc_mode,
        "detach_base": detach_base,
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
    final_gap = float(last["mAP"]) - float(best["mAP"])
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
        "final_minus_best_mAP": f"{final_gap:.1f}",
        "log_file": log_file,
    })

for path, dialect in ((summary_tsv, "excel-tab"), (summary_csv, "excel")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect=dialect)
        writer.writeheader()
        writer.writerows(rows)

print()
print(
    f"{'name':<24} {'status':<12} {'osbbm':<7} {'pamc':<6} {'w':<5} "
    f"{'mode':<10} {'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} "
    f"{'best_ep':<8} {'best_mAP':<8} {'best_R1':<8} {'gap':<6}"
)
for row in rows:
    print(
        f"{row['name']:<24} {row['status']:<12} {row['osbbm_enabled']:<7} "
        f"{row['pamc_enabled']:<6} {row['pamc_weight']:<5} {row['pamc_mode']:<10} "
        f"{row['last_epoch']:<8} {row['last_mAP']:<8} {row['last_R1']:<8} "
        f"{row['best_epoch']:<8} {row['best_mAP']:<8} {row['best_R1']:<8} "
        f"{row['final_minus_best_mAP']:<6}"
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
echo "[PAM-Consistency] All experiments finished."

if [ "${failures}" -ne 0 ]; then
  echo "[PAM-Consistency] One or more experiments failed. Check the logs above and ${OUTPUT_BASE}."
  exit 1
fi
