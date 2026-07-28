#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_hypothesis20}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
EVAL_EPOCHS_CSV="${EVAL_EPOCHS_CSV:-40,80,120,160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a EVAL_EPOCHS <<< "${EVAL_EPOCHS_CSV}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# name|family|os_lr|os_wd|ratr|pair|lambda|intra_target|inter_target|local|osbbm|feature|seed
declare -a EXPERIMENTS=(
  "h20_base_s42|base|1.0|0.05|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_base_s3407|base|1.0|0.05|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_oswd0005_s42|optim|1.0|0.0005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_oswd0005_s3407|optim|1.0|0.0005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_oswd005_s42|optim|1.0|0.005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_oswd005_s3407|optim|1.0|0.005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_oslr05_wd0005_s42|optim|0.5|0.0005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_oslr05_wd0005_s3407|optim|0.5|0.0005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_oslr20_wd0005_s42|optim|2.0|0.0005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_oslr20_wd0005_s3407|optim|2.0|0.0005|False|mamba_osnet|0.0|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_ratr_mo_t5030_s42|ratr|1.0|0.05|True|mamba_osnet|0.05|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_ratr_mo_t5030_s3407|ratr|1.0|0.05|True|mamba_osnet|0.05|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_ratr_mo_t3020_s42|ratr|1.0|0.05|True|mamba_osnet|0.05|0.3|0.2|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_ratr_mo_t3020_s3407|ratr|1.0|0.05|True|mamba_osnet|0.05|0.3|0.2|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_ratr_mf_t5030_s42|ratr|1.0|0.05|True|mamba_fused|0.05|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|42"
  "h20_ratr_mf_t5030_s3407|ratr|1.0|0.05|True|mamba_fused|0.05|0.5|0.3|False|False|weighted_mamba_fdmf_osnet|3407"
  "h20_s3local_s42|local|1.0|0.05|False|mamba_osnet|0.0|0.5|0.3|True|False|weighted_mamba_fdmf_osnet_stage3local|42"
  "h20_s3local_s3407|local|1.0|0.05|False|mamba_osnet|0.0|0.5|0.3|True|False|weighted_mamba_fdmf_osnet_stage3local|3407"
  "h20_osbbm8m2_s42|osbbm|1.0|0.05|False|mamba_osnet|0.0|0.5|0.3|False|True|weighted_mamba_fdmf_osnet|42"
  "h20_osbbm8m2_s3407|osbbm|1.0|0.05|False|mamba_osnet|0.0|0.5|0.3|False|True|weighted_mamba_fdmf_osnet|3407"
)

common_opts() {
  local gpu="$1" os_lr="$2" os_wd="$3" ratr="$4" pair="$5" lambda="$6"
  local intra="$7" inter="$8" local_enabled="$9" osbbm_enabled="${10}"
  local -a pretrain_opts=()
  if [[ -n "${OSNET_PRETRAIN}" ]]; then
    pretrain_opts=(MODEL.OSNET_FUSION.PRETRAIN_PATH "'${OSNET_PRETRAIN}'")
  fi
  printf '%s\n' \
    MODEL.DEVICE_ID "'${gpu}'" \
    MODEL.OSNET_FUSION.ENABLED True \
    MODEL.OSNET_FUSION.OSNET_TYPE "'osnet_x1_0'" \
    MODEL.OSNET_FUSION.FUSION_TYPE "'fdmf'" \
    MODEL.OSNET_FUSION.FUSION_NORM "'none'" \
    MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT 0.5 \
    MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT 1.0 \
    MODEL.OSNET_FUSION.FCU_ENABLED True \
    MODEL.OSNET_FUSION.FCU_STAGES "[2,3]" \
    MODEL.OSNET_FUSION.FCU_DIRECTION "'bidirectional'" \
    MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION "'osnet_to_mamba'" \
    MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION "'mamba_to_osnet'" \
    MODEL.OSNET_FUSION.FDMF_FUSED_FORM "'mamba_fdmf'" \
    MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED True \
    MODEL.OSNET_FUSION.COMPLEMENTARITY.MODE "'none'" \
    MODEL.OSNET_FUSION.PEER_COMPLEMENT.ENABLED False \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${local_enabled}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES 2 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT 0.1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH 1 \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS True \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT 0.3 \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED "${osbbm_enabled}" \
    INPUT.OSBBM.PROB 0.5 \
    INPUT.OSBBM.NUM_BLOCKS 8 \
    INPUT.OSBBM.NUM_MIX_BLOCKS 2 \
    INPUT.OSBBM.GRAY_PROB 0.5 \
    INPUT.OSBBM.APPLY_TO "'base'" \
    INPUT.OSBBM.MIXED_LABEL True \
    INPUT.OSBBM.SCHEDULE "'always'" \
    MODEL.MAMBAVISION.USE_SFM False \
    SOLVER.OSNET_LR_FACTOR "${os_lr}" \
    SOLVER.OSNET_WEIGHT_DECAY "${os_wd}" \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS "${os_wd}" \
    SOLVER.OSNET_FUSION_LR_FACTOR 3.0 \
    SOLVER.RATR_ENABLED "${ratr}" \
    SOLVER.RATR_MODE "'hinge'" \
    SOLVER.RATR_BRANCH_PAIR "'${pair}'" \
    SOLVER.RATR_LAMBDA "${lambda}" \
    SOLVER.RATR_TAU 0.1 \
    SOLVER.RATR_INTRA_TARGET "${intra}" \
    SOLVER.RATR_INTER_TARGET "${inter}" \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.CHECKPOINT_PERIOD 40 \
    SOLVER.EVAL_PERIOD 20 \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_train() {
  local idx="$1" spec="$2"
  local name family os_lr os_wd ratr pair lambda intra inter local_enabled osbbm_enabled feature seed
  IFS='|' read -r name family os_lr os_wd ratr pair lambda intra inter local_enabled osbbm_enabled feature seed <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" output_dir="${OUTPUT_BASE}/${name}"
  local -a opts=()
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then
    echo "[H20] SKIP train ${name}"
    return
  fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${os_lr}" "${os_wd}" "${ratr}" "${pair}" "${lambda}" "${intra}" "${inter}" "${local_enabled}" "${osbbm_enabled}")
  echo "[H20] TRAIN gpu=${gpu} family=${family} exp=${name} os_lr=${os_lr} os_wd=${os_wd} ratr=${ratr} local=${local_enabled} osbbm=${osbbm_enabled}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${seed}" TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2" epoch="$3"
  local name family os_lr os_wd ratr pair lambda intra inter local_enabled osbbm_enabled feature seed
  IFS='|' read -r name family os_lr os_wd ratr pair lambda intra inter local_enabled osbbm_enabled feature seed <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" weight="${OUTPUT_BASE}/${name}/transformer_${epoch}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${epoch}/${name}" test_log="${OUTPUT_BASE}/eval/ep${epoch}/${name}/test_log.txt"
  local -a opts=()
  [[ -f "${weight}" ]] || { echo "[H20] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${test_log}" ]] && grep -q "Rank-10" "${test_log}"; then
    echo "[H20] SKIP eval ${name} ep=${epoch}"
    return
  fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${os_lr}" "${os_wd}" "${ratr}" "${pair}" "${lambda}" "${intra}" "${inter}" "${local_enabled}" "${osbbm_enabled}")
  echo "[H20] EVAL gpu=${gpu} exp=${name} ep=${epoch} feature=${feature}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${seed}" TEST.WEIGHT "'${weight}'" \
    TEST.FEAT_MODE "'${feature}'" TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
}

run_pool() {
  local action="$1" running=0 failures=0 idx=0 spec epoch
  for spec in "${EXPERIMENTS[@]}"; do
    if [[ "${action}" == "train" ]]; then
      run_train "${idx}" "${spec}" & idx=$((idx + 1)); running=$((running + 1))
    else
      for epoch in "${EVAL_EPOCHS[@]}"; do
        run_eval "${idx}" "${spec}" "${epoch}" & idx=$((idx + 1)); running=$((running + 1))
        if [[ "${running}" -ge "${MAX_JOBS}" ]]; then wait -n || failures=1; running=$((running - 1)); fi
      done
      continue
    fi
    if [[ "${running}" -ge "${MAX_JOBS}" ]]; then wait -n || failures=1; running=$((running - 1)); fi
  done
  while [[ "${running}" -gt 0 ]]; do wait -n || failures=1; running=$((running - 1)); done
  return "${failures}"
}

summarize() {
  local specs="$(printf '%s\n' "${EXPERIMENTS[@]}")"
  H20_SPECS="${specs}" python - "${OUTPUT_BASE}" "${EVAL_EPOCHS_CSV}" <<'PY'
import os, re, statistics, sys
from collections import defaultdict
base, epochs_csv = sys.argv[1:3]
epochs = [int(x) for x in epochs_csv.split(',') if x]
specs = [x for x in os.environ.get('H20_SPECS', '').splitlines() if x]
map_re = re.compile(r'\bmAP:\s*([0-9.]+)%')
rank_re = re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')
def parse(path):
    out = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = map_re.search(line)
            if m: out = {'mAP': float(m.group(1))}; continue
            m = rank_re.search(line)
            if m and 'mAP' in out: out['R' + m.group(1)] = float(m.group(2))
    return out if all(k in out for k in ('mAP','R1','R5','R10')) else None
rows=[]
for spec in specs:
    name,family,oslr,oswd,ratr,pair,lam,it,et,local,osbbm,feature,seed=spec.split('|')
    config=name.rsplit('_s',1)[0]
    for ep in epochs:
        path=os.path.join(base,'eval',f'ep{ep}',name,'test_log.txt')
        rows.append((name,config,family,ep,seed,oslr,oswd,parse(path) if os.path.exists(path) else None))
print(f"{'experiment':<31} {'family':<7} {'ep':>4} {'seed':>5} {'lr':>4} {'wd':>7} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
for name,_,family,ep,seed,lr,wd,res in rows:
    v=['NA']*4 if res is None else [f"{res[k]:.1f}" for k in ('mAP','R1','R5','R10')]
    print(f"{name:<31} {family:<7} {ep:>4} {seed:>5} {lr:>4} {wd:>7} {v[0]:>6} {v[1]:>6} {v[2]:>6} {v[3]:>6}")
groups=defaultdict(list)
for _,config,family,ep,_,_,_,res in rows:
    if res: groups[(config,family,ep)].append(res)
print('\nconfiguration means')
final=[]
for (config,family,ep),results in sorted(groups.items()):
    mm=statistics.mean(x['mAP'] for x in results); rr=statistics.mean(x['R1'] for x in results)
    print(f"{config:<27} {family:<7} ep={ep:>3} n={len(results)} mAP={mm:.2f} R1={rr:.2f}")
    if ep==160 and len(results)>=2: final.append((mm,rr,config,family))
if final:
    best=max(final,key=lambda x:(x[0],x[1]))
    print(f"\nbest_final: {best[2]} family={best[3]} mean_mAP={best[0]:.2f} mean_R1={best[1]:.2f}")
PY
}

echo "[H20] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/summary.txt"
echo "[H20] Summary saved to ${OUTPUT_BASE}/summary.txt"
