#!/usr/bin/env bash
# Run LocalJPM branch ablations, two experiments at a time by default.
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

CONFIG="${CONFIG:-configs/OCC_Duke/mambavision_tiny_transreid_pam_padeaug_localjpm_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/localjpm_ablation}"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "ljpm_k4_shuffle_bimamba_d2_id020_tri005_gs02|4|shuffle|5|2|bimamba|2|256|0.20|0.05|10|0.00|0.20|global|0.8"
  "ljpm_k4_shuffle_attn_d2_id020_tri005_gs02|4|shuffle|5|2|attn|2|256|0.20|0.05|10|0.00|0.20|global|0.8"
  "ljpm_k4_shuffle_mvmixer_d2_id020_tri005_gs02|4|shuffle|5|2|mvmixer|2|256|0.20|0.05|10|0.00|0.20|global|0.8"
  "ljpm_k4_noshuffle_bimamba_d2_id020_tri005_gs02|4|noshuffle|0|1|bimamba|2|256|0.20|0.05|10|0.00|0.20|global|0.8"
  "ljpm_k4_shuffle_bimamba_d1_id020_tri005_gs02|4|shuffle|5|2|bimamba|1|256|0.20|0.05|10|0.00|0.20|global|0.8"
  "ljpm_k4_shuffle_bimamba_d3_id020_tri005_gs02|4|shuffle|5|2|bimamba|3|256|0.20|0.05|10|0.00|0.20|global|0.8"
  "ljpm_k4_shuffle_bimamba_d2_id020_tri000_gs02|4|shuffle|5|2|bimamba|2|256|0.20|0.00|10|0.00|0.20|global|0.8"
  "ljpm_k4_shuffle_bimamba_d2_id020_tri005_gs01|4|shuffle|5|2|bimamba|2|256|0.20|0.05|10|0.00|0.10|global|0.8"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name parts group_mode shift shuffle_groups refiner depth feat_dim id_weight tri_weight tri_warmup dissimilar_weight grad_scale inference local_scale

  IFS='|' read -r name parts group_mode shift shuffle_groups refiner depth feat_dim id_weight tri_weight tri_warmup dissimilar_weight grad_scale inference local_scale <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"

  echo "[LocalJPM] GPU=${gpu} EXP=${name} parts=${parts} group=${group_mode} refiner=${refiner} depth=${depth} id=${id_weight} tri=${tri_weight} grad=${grad_scale} infer=${inference}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.STRIPE_AUX.ENABLED False \
    MODEL.LOCAL_CLS.ENABLED False \
    MODEL.LOCAL_JPM.ENABLED True \
    MODEL.LOCAL_JPM.NUM_PARTS "${parts}" \
    MODEL.LOCAL_JPM.GROUP_MODE "'${group_mode}'" \
    MODEL.LOCAL_JPM.SHIFT "${shift}" \
    MODEL.LOCAL_JPM.SHUFFLE_GROUPS "${shuffle_groups}" \
    MODEL.LOCAL_JPM.REFINER "'${refiner}'" \
    MODEL.LOCAL_JPM.DEPTH "${depth}" \
    MODEL.LOCAL_JPM.FEAT_DIM "${feat_dim}" \
    MODEL.LOCAL_JPM.ID_WEIGHT "${id_weight}" \
    MODEL.LOCAL_JPM.TRI_WEIGHT "${tri_weight}" \
    MODEL.LOCAL_JPM.TRI_WARMUP_EPOCHS "${tri_warmup}" \
    MODEL.LOCAL_JPM.DISSIMILAR_WEIGHT "${dissimilar_weight}" \
    MODEL.LOCAL_JPM.GRAD_SCALE "${grad_scale}" \
    MODEL.LOCAL_JPM.INFERENCE "'${inference}'" \
    MODEL.LOCAL_JPM.LOCAL_SCALE "${local_scale}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local summary_file="${OUTPUT_BASE}/summary.tsv"
  local summary_csv="${OUTPUT_BASE}/summary.csv"
  local specs

  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[LocalJPM] Summarizing results -> ${summary_file} and ${summary_csv}"

  LOCALJPM_SPECS="${specs}" python - "${OUTPUT_BASE}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("LOCALJPM_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "parts", "group_mode", "shift", "shuffle_groups", "refiner", "depth",
    "feat_dim", "id_weight", "tri_weight", "tri_warmup", "dissimilar_weight", "grad_scale",
    "inference", "local_scale",
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
    (
        name,
        parts,
        group_mode,
        shift,
        shuffle_groups,
        refiner,
        depth,
        feat_dim,
        id_weight,
        tri_weight,
        tri_warmup,
        dissimilar_weight,
        grad_scale,
        inference,
        local_scale,
    ) = spec.split("|")
    output_dir = os.path.join(output_base, name)
    train_log = os.path.join(output_dir, "train_log.txt")
    test_log = os.path.join(output_dir, "test_log.txt")
    log_file = train_log if os.path.exists(train_log) else test_log
    base = {
        "name": name,
        "parts": parts,
        "group_mode": group_mode,
        "shift": shift,
        "shuffle_groups": shuffle_groups,
        "refiner": refiner,
        "depth": depth,
        "feat_dim": feat_dim,
        "id_weight": id_weight,
        "tri_weight": tri_weight,
        "tri_warmup": tri_warmup,
        "dissimilar_weight": dissimilar_weight,
        "grad_scale": grad_scale,
        "inference": inference,
        "local_scale": local_scale,
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
    f"{'name':<48} {'status':<12} {'refiner':<9} {'depth':<5} {'group':<9} {'gs':<5} "
    f"{'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} "
    f"{'best_ep':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<48} {row['status']:<12} {row['refiner']:<9} {row['depth']:<5} {row['group_mode']:<9} {row['grad_scale']:<5} "
        f"{row['last_epoch']:<8} {row['last_mAP']:<8} {row['last_R1']:<8} "
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
echo "[LocalJPM] All experiments finished."

if [ "${failures}" -ne 0 ]; then
  echo "[LocalJPM] One or more experiments failed. Check the logs above and ${OUTPUT_BASE}."
  exit 1
fi
