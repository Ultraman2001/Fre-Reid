#!/usr/bin/env bash
# Evaluate all LocalJPM ablation checkpoints with concat and late distance fusion.
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
TRAIN_BASE="${TRAIN_BASE:-./logs/OCC-Duke/localjpm_ablation}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/OCC-Duke/localjpm_concat_eval}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-160}"
WEIGHT_NAME="${WEIGHT_NAME:-transformer_${CHECKPOINT_EPOCH}.pth}"
FUSION_ALPHAS="${FUSION_ALPHAS:-0.02,0.05,0.10,0.20}"
FUSION_ALPHA_LIST="[${FUSION_ALPHAS}]"

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

run_eval() {
  local idx="$1"
  local spec="$2"
  local name parts group_mode shift shuffle_groups refiner depth feat_dim id_weight tri_weight tri_warmup dissimilar_weight grad_scale inference local_scale

  IFS='|' read -r name parts group_mode shift shuffle_groups refiner depth feat_dim id_weight tri_weight tri_warmup dissimilar_weight grad_scale inference local_scale <<< "${spec}"

  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local weight="${TRAIN_BASE}/${name}/${WEIGHT_NAME}"
  local output_dir="${OUTPUT_BASE}/${name}"

  mkdir -p "${output_dir}"

  if [ ! -f "${weight}" ]; then
    echo "[LocalJPMEval] SKIP missing checkpoint: ${weight}"
    return 0
  fi

  echo "[LocalJPMEval] GPU=${gpu} EXP=${name} weight=${weight} infer=concat"

  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    TEST.WEIGHT "${weight}" \
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
    MODEL.LOCAL_JPM.INFERENCE "'concat'" \
    MODEL.LOCAL_JPM.LOCAL_SCALE "${local_scale}" \
    TEST.LOCAL_DIST_FUSION.ENABLED True \
    TEST.LOCAL_DIST_FUSION.GLOBAL_BRANCH "'backbone'" \
    TEST.LOCAL_DIST_FUSION.LOCAL_BRANCH "'fused'" \
    TEST.LOCAL_DIST_FUSION.ALPHAS "${FUSION_ALPHA_LIST}" \
    OUTPUT_DIR "${output_dir}"
}

summarize_results() {
  local summary_file="${OUTPUT_BASE}/summary.tsv"
  local summary_csv="${OUTPUT_BASE}/summary.csv"
  local specs

  specs="$(printf "%s\n" "${EXPERIMENTS[@]}")"
  echo "[LocalJPMEval] Summarizing results -> ${summary_file} and ${summary_csv}"

  LOCALJPM_EVAL_SPECS="${specs}" LOCALJPM_FUSION_ALPHAS="${FUSION_ALPHAS}" python - "${TRAIN_BASE}" "${OUTPUT_BASE}" "${WEIGHT_NAME}" "${summary_file}" "${summary_csv}" <<'PY'
import csv
import os
import re
import sys

train_base, output_base, weight_name, summary_tsv, summary_csv = sys.argv[1:6]
specs = [line for line in os.environ.get("LOCALJPM_EVAL_SPECS", "").splitlines() if line.strip()]
fusion_alphas = [
    float(item)
    for item in os.environ.get("LOCALJPM_FUSION_ALPHAS", "0.02,0.05,0.10,0.20").split(",")
    if item.strip()
]

def alpha_tag(alpha):
    text = f"{float(alpha):.4f}".rstrip("0").rstrip(".")
    return "a" + text.replace(".", "p")

branches = ["backbone", "fused", "concat"]
branches.extend([f"late_fusion_{alpha_tag(alpha)}" for alpha in fusion_alphas])
metric_fields = []
for branch in branches:
    metric_fields.extend([f"{branch}_mAP", f"{branch}_R1", f"{branch}_R5", f"{branch}_R10"])

fields = [
    "name", "status", "parts", "group_mode", "shift", "shuffle_groups", "refiner", "depth",
    "feat_dim", "id_weight", "tri_weight", "tri_warmup", "dissimilar_weight", "grad_scale",
    "weight_file",
    *metric_fields,
    "log_file",
]

branch_re = re.compile(r"===\s+([A-Za-z0-9_]+)\s+Results\s+===")
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
        _inference,
        _local_scale,
    ) = spec.split("|")
    weight_file = os.path.join(train_base, name, weight_name)
    log_file = os.path.join(output_base, name, "test_log.txt")
    row = {
        "name": name,
        "status": "ok",
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
header = (
    f"{'name':<48} {'status':<14} {'refiner':<9} {'depth':<5} {'group':<9} {'gs':<5} "
    f"{'B_mAP':<7} {'B_R1':<7} {'F_mAP':<7} {'F_R1':<7} {'C_mAP':<7} {'C_R1':<7}"
)
for alpha in fusion_alphas:
    tag = alpha_tag(alpha)
    header += f" {tag}_mAP".ljust(9) + f" {tag}_R1".ljust(8)
print(header)
for row in rows:
    line = (
        f"{row['name']:<48} {row['status']:<14} {row['refiner']:<9} {row['depth']:<5} {row['group_mode']:<9} {row['grad_scale']:<5} "
        f"{row['backbone_mAP']:<7} {row['backbone_R1']:<7} "
        f"{row['fused_mAP']:<7} {row['fused_R1']:<7} "
        f"{row['concat_mAP']:<7} {row['concat_R1']:<7}"
    )
    for alpha in fusion_alphas:
        branch = f"late_fusion_{alpha_tag(alpha)}"
        line += f" {row[f'{branch}_mAP']:<8} {row[f'{branch}_R1']:<7}"
    print(line)
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
echo "[LocalJPMEval] All evaluations finished."

if [ "${failures}" -ne 0 ]; then
  echo "[LocalJPMEval] One or more evaluations failed. Check logs under ${OUTPUT_BASE}."
  exit 1
fi
