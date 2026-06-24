#!/usr/bin/env bash
# Run StripeTokenContext ablations, two experiments at a time by default.
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

CONFIG="${CONFIG:-configs/OCC_Duke/mambavision_tiny_transreid_pam_padeaug_stripectx_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/stripectx_ablation}"

mkdir -p "${OUTPUT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=("0")
fi

declare -a EXPERIMENTS=(
  "s2_ctx_d256_tri_w100|2|gem|256|1.00|True|True|True|global|1.0"
  "s2_ctx_d256_tri_w050|2|gem|256|0.50|True|True|True|global|1.0"
  "s2_ctx_d256_tri_w030|2|gem|256|0.30|True|True|True|global|1.0"
  "s2_ctx_d256_tri_w020|2|gem|256|0.20|True|True|True|global|1.0"
  "s4_ctx_d256_tri_w100|4|gem|256|1.00|True|True|True|global|1.0"
  "s4_ctx_d256_tri_w050|4|gem|256|0.50|True|True|True|global|1.0"
  "s4_ctx_d256_tri_w030|4|gem|256|0.30|True|True|True|global|1.0"
  "s4_ctx_d256_tri_w020|4|gem|256|0.20|True|True|True|global|1.0"
  "s2_raw_d256_tri_w100|2|gem|256|1.00|True|False|False|global|1.0"
  "s4_raw_d256_tri_w100|4|gem|256|1.00|True|False|False|global|1.0"
  "s2_uni_d256_tri_w100|2|gem|256|1.00|True|True|False|global|1.0"
  "s2_ctx_d512_tri_w100|2|gem|512|1.00|True|True|True|global|1.0"
  "s2_ctx_d256_id_w100|2|gem|256|1.00|False|True|True|global|1.0"
)

run_experiment() {
  local idx="$1"
  local spec="$2"
  local name num_stripes pooling feat_dim loss_weight use_triplet token_context bidirectional inference local_scale

  IFS='|' read -r name num_stripes pooling feat_dim loss_weight use_triplet token_context bidirectional inference local_scale <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local output_dir="${OUTPUT_BASE}/${name}"

  echo "[StripeCtx] GPU=${gpu} EXP=${name} stripes=${num_stripes} pool=${pooling} dim=${feat_dim} weight=${loss_weight} triplet=${use_triplet} ctx=${token_context} bi=${bidirectional} infer=${inference}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.STRIPE_AUX.ENABLED True \
    MODEL.STRIPE_AUX.NUM_STRIPES "${num_stripes}" \
    MODEL.STRIPE_AUX.POOLING_TYPE "'${pooling}'" \
    MODEL.STRIPE_AUX.FEAT_DIM "${feat_dim}" \
    MODEL.STRIPE_AUX.LOSS_WEIGHT "${loss_weight}" \
    MODEL.STRIPE_AUX.USE_TRIPLET "${use_triplet}" \
    MODEL.STRIPE_AUX.INFERENCE "'${inference}'" \
    MODEL.STRIPE_AUX.LOCAL_SCALE "${local_scale}" \
    MODEL.STRIPE_AUX.TOKEN_CONTEXT.ENABLED "${token_context}" \
    MODEL.STRIPE_AUX.TOKEN_CONTEXT.BIDIRECTIONAL "${bidirectional}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local summary_file="${OUTPUT_BASE}/summary.tsv"
  local summary_csv="${OUTPUT_BASE}/summary.csv"
  local specs

  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[StripeCtx] Summarizing results -> ${summary_file} and ${summary_csv}"

  STRIPECTX_SPECS="${specs}" python - "${OUTPUT_BASE}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

output_base, summary_tsv, summary_csv = sys.argv[1:4]
specs = [line for line in os.environ.get("STRIPECTX_SPECS", "").splitlines() if line.strip()]

fields = [
    "name", "status", "stripes", "pooling", "feat_dim", "loss_weight",
    "triplet", "token_context", "bidirectional", "inference", "local_scale",
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
        stripes,
        pooling,
        feat_dim,
        loss_weight,
        triplet,
        token_context,
        bidirectional,
        inference,
        local_scale,
    ) = spec.split("|")
    output_dir = os.path.join(output_base, name)
    train_log = os.path.join(output_dir, "train_log.txt")
    test_log = os.path.join(output_dir, "test_log.txt")
    log_file = train_log if os.path.exists(train_log) else test_log
    base = {
        "name": name,
        "stripes": stripes,
        "pooling": pooling,
        "feat_dim": feat_dim,
        "loss_weight": loss_weight,
        "triplet": triplet,
        "token_context": token_context,
        "bidirectional": bidirectional,
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
    f"{'name':<24} {'status':<12} {'stripes':<7} {'w':<6} {'tri':<6} {'ctx':<6} {'bi':<6} "
    f"{'last_ep':<8} {'last_mAP':<8} {'last_R1':<8} "
    f"{'best_ep':<8} {'best_mAP':<8} {'best_R1':<8}"
)
for row in rows:
    print(
        f"{row['name']:<24} {row['status']:<12} {row['stripes']:<7} {row['loss_weight']:<6} "
        f"{row['triplet']:<6} {row['token_context']:<6} {row['bidirectional']:<6} "
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
echo "[StripeCtx] All experiments finished."

if [ "${failures}" -ne 0 ]; then
  echo "[StripeCtx] One or more experiments failed. Check the logs above and ${OUTPUT_BASE}."
  exit 1
fi
