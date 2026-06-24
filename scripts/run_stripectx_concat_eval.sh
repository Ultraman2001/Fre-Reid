#!/usr/bin/env bash
# Evaluate all StripeTokenContext ablation checkpoints with concat enabled.
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
TRAIN_BASE="${TRAIN_BASE:-./logs/OCC-Duke/stripectx_ablation}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/stripectx_concat_eval}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-160}"
WEIGHT_NAME="${WEIGHT_NAME:-transformer_${CHECKPOINT_EPOCH}.pth}"

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

run_eval() {
  local idx="$1"
  local spec="$2"
  local name num_stripes pooling feat_dim loss_weight use_triplet token_context bidirectional inference local_scale

  IFS='|' read -r name num_stripes pooling feat_dim loss_weight use_triplet token_context bidirectional inference local_scale <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${TRAIN_BASE}/${name}/${WEIGHT_NAME}"
  local output_dir="${OUTPUT_BASE}/${name}"

  mkdir -p "${output_dir}"

  if [ ! -f "${weight}" ]; then
    echo "[StripeCtxEval] SKIP missing checkpoint: ${weight}"
    return 0
  fi

  echo "[StripeCtxEval] GPU=${gpu} EXP=${name} weight=${weight} infer=concat"

  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    TEST.WEIGHT "${weight}" \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.STRIPE_AUX.ENABLED True \
    MODEL.STRIPE_AUX.NUM_STRIPES "${num_stripes}" \
    MODEL.STRIPE_AUX.POOLING_TYPE "'${pooling}'" \
    MODEL.STRIPE_AUX.FEAT_DIM "${feat_dim}" \
    MODEL.STRIPE_AUX.LOSS_WEIGHT "${loss_weight}" \
    MODEL.STRIPE_AUX.USE_TRIPLET "${use_triplet}" \
    MODEL.STRIPE_AUX.INFERENCE "'concat'" \
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
  echo "[StripeCtxEval] Summarizing results -> ${summary_file} and ${summary_csv}"

  STRIPECTX_EVAL_SPECS="${specs}" python - "${TRAIN_BASE}" "${OUTPUT_BASE}" "${WEIGHT_NAME}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

train_base, output_base, weight_name, summary_tsv, summary_csv = sys.argv[1:6]
specs = [line for line in os.environ.get("STRIPECTX_EVAL_SPECS", "").splitlines() if line.strip()]

branches = ("backbone", "fused", "concat")
metric_fields = []
for branch in branches:
    metric_fields.extend([f"{branch}_mAP", f"{branch}_R1", f"{branch}_R5", f"{branch}_R10"])

fields = [
    "name", "status", "stripes", "pooling", "feat_dim", "loss_weight",
    "triplet", "token_context", "bidirectional", "weight_file",
    *metric_fields,
    "log_file",
]

branch_re = re.compile(r"===\s+(BACKBONE|FUSED|CONCAT)\s+Results\s+===")
map_re = re.compile(r"\bmAP:\s*([0-9.]+)%")
rank_re = re.compile(r"Rank-(1|5|10)\s*:?\s*([0-9.]+)%")


def parse_log(log_path):
    results = {}
    current = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            branch_match = branch_re.search(line)
            if branch_match:
                current = branch_match.group(1).lower()
                results.setdefault(current, {})
                continue
            if current is None:
                continue
            map_match = map_re.search(line)
            if map_match:
                results[current]["mAP"] = map_match.group(1)
                continue
            rank_match = rank_re.search(line)
            if rank_match:
                rank, value = rank_match.groups()
                results[current][f"R{rank}"] = value
    return results


def empty_metrics(row):
    for branch in branches:
        row[f"{branch}_mAP"] = "NA"
        row[f"{branch}_R1"] = "NA"
        row[f"{branch}_R5"] = "NA"
        row[f"{branch}_R10"] = "NA"


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
        _inference,
        _local_scale,
    ) = spec.split("|")

    weight_file = os.path.join(train_base, name, weight_name)
    log_file = os.path.join(output_base, name, "test_log.txt")
    row = {
        "name": name,
        "status": "ok",
        "stripes": stripes,
        "pooling": pooling,
        "feat_dim": feat_dim,
        "loss_weight": loss_weight,
        "triplet": triplet,
        "token_context": token_context,
        "bidirectional": bidirectional,
        "weight_file": weight_file,
        "log_file": log_file,
    }
    empty_metrics(row)

    if not os.path.exists(weight_file):
        row["status"] = "missing_weight"
        row["log_file"] = "NA"
        rows.append(row)
        continue
    if not os.path.exists(log_file):
        row["status"] = "missing_log"
        rows.append(row)
        continue

    parsed = parse_log(log_file)
    if not parsed:
        row["status"] = "no_eval_found"
        rows.append(row)
        continue

    for branch in branches:
        metrics = parsed.get(branch, {})
        row[f"{branch}_mAP"] = metrics.get("mAP", "NA")
        row[f"{branch}_R1"] = metrics.get("R1", "NA")
        row[f"{branch}_R5"] = metrics.get("R5", "NA")
        row[f"{branch}_R10"] = metrics.get("R10", "NA")
    rows.append(row)

for path, dialect in ((summary_tsv, "excel-tab"), (summary_csv, "excel")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect=dialect)
        writer.writeheader()
        writer.writerows(rows)

print()
print(
    f"{'name':<24} {'status':<14} {'B_mAP':<7} {'B_R1':<7} "
    f"{'F_mAP':<7} {'F_R1':<7} {'C_mAP':<7} {'C_R1':<7}"
)
for row in rows:
    print(
        f"{row['name']:<24} {row['status']:<14} "
        f"{row['backbone_mAP']:<7} {row['backbone_R1']:<7} "
        f"{row['fused_mAP']:<7} {row['fused_R1']:<7} "
        f"{row['concat_mAP']:<7} {row['concat_R1']:<7}"
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
  run_eval "${idx}" "${EXPERIMENTS[$idx]}" &
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
echo "[StripeCtxEval] All evaluations finished."

if [ "${failures}" -ne 0 ]; then
  echo "[StripeCtxEval] One or more evaluations failed. Check logs under ${OUTPUT_BASE}."
  exit 1
fi
