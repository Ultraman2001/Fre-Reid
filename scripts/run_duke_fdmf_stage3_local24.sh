#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-all}"
GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_JOBS="${2:-1}"
CONFIG="${CONFIG:-configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml}"
OUTPUT_BASE="${OUTPUT_BASE:-./logs/Duke/fdmf_stage3_local24}"
OSNET_PRETRAIN="${OSNET_PRETRAIN:-/workspace/pretrained/osnet_x1_0_imagenet.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
EVAL_EPOCHS_CSV="${EVAL_EPOCHS_CSV:-40,80,120,160}"
TEST_BATCH="${TEST_BATCH:-128}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEARCH_SEED="${SEARCH_SEED:-42}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
SUMMARY_NAME="${SUMMARY_NAME:-summary_s${SEARCH_SEED}.txt}"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" && "${MODE}" != "summary" ]]; then
  echo "MODE must be train, eval, all, or summary" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a EVAL_EPOCHS <<< "${EVAL_EPOCHS_CSV}"
[[ "${#GPUS[@]}" -gt 0 ]] || GPUS=("0")

# Exploration uses one seed. Re-run only shortlisted configurations with
# SEARCH_SEED=3407 and EXPERIMENT_FILTER=name1,name2.
# name|family|enabled|type|parts|part_dim|source|interaction|depth|share|temperature|prior|balance_w|order_w|loss_w|infer_w|seed
declare -a EXPERIMENTS=(
  "l24_nolocal_s${SEARCH_SEED}|control|False|hard|2|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_hard_k2_d0_s${SEARCH_SEED}|hard|True|hard|2|0|osnet|hadamard|0|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_hard_k2_d1_s${SEARCH_SEED}|hard|True|hard|2|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_hard_k2_d2_s${SEARCH_SEED}|hard|True|hard|2|0|osnet|hadamard|2|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_hard_k3_d1_s${SEARCH_SEED}|hard|True|hard|3|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_hard_k4_d1_s${SEARCH_SEED}|hard|True|hard|4|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_hard_k2_unshared_s${SEARCH_SEED}|hard|True|hard|2|0|osnet|hadamard|1|False|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_d0_s${SEARCH_SEED}|depth|True|soft|2|0|osnet|hadamard|0|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_d1_s${SEARCH_SEED}|soft|True|soft|2|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_d2_s${SEARCH_SEED}|depth|True|soft|2|0|osnet|hadamard|2|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k3_d1_s${SEARCH_SEED}|parts|True|soft|3|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k4_d1_s${SEARCH_SEED}|parts|True|soft|4|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_concat_s${SEARCH_SEED}|interaction|True|soft|2|0|osnet|concat|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_fused_k2_had_s${SEARCH_SEED}|source|True|soft|2|0|fused|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_mamba_k2_had_s${SEARCH_SEED}|source|True|soft|2|0|mamba|hadamard|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_fused_k2_concat_s${SEARCH_SEED}|source_inter|True|soft|2|0|fused|concat|1|True|1.0|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_t05_s${SEARCH_SEED}|temperature|True|soft|2|0|osnet|hadamard|1|True|0.5|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_t15_s${SEARCH_SEED}|temperature|True|soft|2|0|osnet|hadamard|1|True|1.5|1.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_prior0_s${SEARCH_SEED}|prior|True|soft|2|0|osnet|hadamard|1|True|1.0|0.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_prior05_s${SEARCH_SEED}|prior|True|soft|2|0|osnet|hadamard|1|True|1.0|0.5|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_prior2_s${SEARCH_SEED}|prior|True|soft|2|0|osnet|hadamard|1|True|1.0|2.0|0.01|0.01|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_noreg_s${SEARCH_SEED}|regular|True|soft|2|0|osnet|hadamard|1|True|1.0|1.0|0.0|0.0|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_balonly_s${SEARCH_SEED}|regular|True|soft|2|0|osnet|hadamard|1|True|1.0|1.0|0.01|0.0|0.1|0.3|${SEARCH_SEED}"
  "l24_soft_os_k2_ordonly_s${SEARCH_SEED}|regular|True|soft|2|0|osnet|hadamard|1|True|1.0|1.0|0.0|0.01|0.1|0.3|${SEARCH_SEED}"
)

if [[ -n "${EXPERIMENT_FILTER}" ]]; then
  declare -a FILTERED=()
  IFS=',' read -r -a FILTERS <<< "${EXPERIMENT_FILTER}"
  for spec in "${EXPERIMENTS[@]}"; do
    name="${spec%%|*}"
    for pattern in "${FILTERS[@]}"; do
      if [[ "${name}" == *"${pattern}"* ]]; then FILTERED+=("${spec}"); break; fi
    done
  done
  EXPERIMENTS=("${FILTERED[@]}")
fi
[[ "${#EXPERIMENTS[@]}" -gt 0 ]] || { echo "No experiment matched EXPERIMENT_FILTER=${EXPERIMENT_FILTER}" >&2; exit 2; }

common_opts() {
  local gpu="$1" enabled="$2" local_type="$3" parts="$4" part_dim="$5"
  local mask_source="$6" interaction="$7" depth="$8" share="$9" temperature="${10}"
  local prior="${11}" balance_w="${12}" order_w="${13}" loss_w="${14}" infer_w="${15}"
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
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED "${enabled}" \
    MODEL.OSNET_FUSION.STAGE3_LOCAL_TYPE "'${local_type}'" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES "${parts}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM "${part_dim}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT "${loss_w}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT "${infer_w}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS "${share}" \
    MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH "${depth}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_MASK_SOURCE "'${mask_source}'" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_INTERACTION "'${interaction}'" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_TEMPERATURE "${temperature}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_PRIOR_SCALE "${prior}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_MARGIN 0.15 \
    MODEL.OSNET_FUSION.STAGE3_SOFT_BALANCE_WEIGHT "${balance_w}" \
    MODEL.OSNET_FUSION.STAGE3_SOFT_ORDER_WEIGHT "${order_w}" \
    INPUT.PAM.ENABLED False \
    INPUT.OSBBM.ENABLED False \
    MODEL.MAMBAVISION.USE_SFM False \
    SOLVER.OSNET_LR_FACTOR 2.0 \
    SOLVER.OSNET_WEIGHT_DECAY 0.0005 \
    SOLVER.OSNET_WEIGHT_DECAY_BIAS 0.0005 \
    SOLVER.OSNET_FUSION_LR_FACTOR 3.0 \
    SOLVER.RATR_ENABLED False \
    SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
    SOLVER.CHECKPOINT_PERIOD 40 \
    SOLVER.EVAL_PERIOD 20 \
    TEST.EVAL_ALL_FEATS False \
    "${pretrain_opts[@]}"
}

run_train() {
  local idx="$1" spec="$2"
  local name family enabled local_type parts part_dim source interaction depth share temperature prior balance_w order_w loss_w infer_w seed
  IFS='|' read -r name family enabled local_type parts part_dim source interaction depth share temperature prior balance_w order_w loss_w infer_w seed <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" output_dir="${OUTPUT_BASE}/${name}"
  local feature='weighted_mamba_fdmf_osnet_stage3local' opts=()
  [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/transformer_${MAX_EPOCHS}.pth" ]]; then echo "[L24] SKIP train ${name}"; return; fi
  mkdir -p "${output_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${local_type}" "${parts}" "${part_dim}" "${source}" "${interaction}" "${depth}" "${share}" "${temperature}" "${prior}" "${balance_w}" "${order_w}" "${loss_w}" "${infer_w}")
  echo "[L24] TRAIN gpu=${gpu} family=${family} exp=${name} type=${local_type} parts=${parts} depth=${depth} source=${source} interaction=${interaction}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${seed}" TEST.FEAT_MODE "'${feature}'" OUTPUT_DIR "${output_dir}"
}

run_eval() {
  local idx="$1" spec="$2" epoch="$3"
  local name family enabled local_type parts part_dim source interaction depth share temperature prior balance_w order_w loss_w infer_w seed
  IFS='|' read -r name family enabled local_type parts part_dim source interaction depth share temperature prior balance_w order_w loss_w infer_w seed <<< "${spec}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}" weight="${OUTPUT_BASE}/${name}/transformer_${epoch}.pth"
  local eval_dir="${OUTPUT_BASE}/eval/ep${epoch}/${name}" feature='weighted_mamba_fdmf_osnet_stage3local' opts=()
  [[ "${enabled}" == "True" ]] || feature='weighted_mamba_fdmf_osnet'
  [[ -f "${weight}" ]] || { echo "[L24] MISSING ${weight}"; return; }
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/test_log.txt" ]] && grep -q "Rank-10" "${eval_dir}/test_log.txt"; then echo "[L24] SKIP eval ${name} ep=${epoch}"; return; fi
  mkdir -p "${eval_dir}"
  mapfile -t opts < <(common_opts "${gpu}" "${enabled}" "${local_type}" "${parts}" "${part_dim}" "${source}" "${interaction}" "${depth}" "${share}" "${temperature}" "${prior}" "${balance_w}" "${order_w}" "${loss_w}" "${infer_w}")
  echo "[L24] EVAL gpu=${gpu} exp=${name} ep=${epoch} feature=${feature}"
  CUDA_VISIBLE_DEVICES="${gpu}" python test.py --config_file "${CONFIG}" \
    "${opts[@]}" SOLVER.SEED "${seed}" TEST.WEIGHT "'${weight}'" TEST.FEAT_MODE "'${feature}'" \
    TEST.NECK_FEAT "'before'" TEST.FEAT_NORM "'yes'" TEST.IMS_PER_BATCH "${TEST_BATCH}" OUTPUT_DIR "${eval_dir}"
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
  L24_SPECS="${specs}" python - "${OUTPUT_BASE}" "${EVAL_EPOCHS_CSV}" <<'PY'
import os, re, statistics, sys
from collections import defaultdict
base, epochs_csv = sys.argv[1:3]
epochs=[int(x) for x in epochs_csv.split(',') if x]
specs=[x for x in os.environ.get('L24_SPECS','').splitlines() if x]
map_re=re.compile(r'\bmAP:\s*([0-9.]+)%'); rank_re=re.compile(r'Rank-(1|5|10)\s*:?[ ]*([0-9.]+)%')
def parse(path):
    out={}
    with open(path,encoding='utf-8',errors='ignore') as f:
        for line in f:
            m=map_re.search(line)
            if m: out={'mAP':float(m.group(1))}; continue
            m=rank_re.search(line)
            if m and 'mAP' in out: out['R'+m.group(1)]=float(m.group(2))
    return out if all(k in out for k in ('mAP','R1','R5','R10')) else None
rows=[]
for spec in specs:
    name,family,en,typ,k,dim,src,inter,depth,share,temp,prior,bw,ow,lw,iw,seed=spec.split('|'); config=name.rsplit('_s',1)[0]
    for ep in epochs:
        path=os.path.join(base,'eval',f'ep{ep}',name,'test_log.txt')
        rows.append((name,config,family,typ,k,src,inter,depth,temp,prior,ep,seed,parse(path) if os.path.exists(path) else None))
print(f"{'experiment':<38} {'fam':<12} {'type':<5} {'k':>2} {'d':>2} {'source':<6} {'inter':<8} {'temp':>4} {'prior':>5} {'ep':>4} {'seed':>5} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}")
for name,_,fam,typ,k,src,inter,depth,temp,prior,ep,seed,res in rows:
    v=['NA']*4 if res is None else [f"{res[x]:.1f}" for x in ('mAP','R1','R5','R10')]
    print(f"{name:<38} {fam:<12} {typ:<5} {k:>2} {depth:>2} {src:<6} {inter:<8} {temp:>4} {prior:>5} {ep:>4} {seed:>5} {v[0]:>6} {v[1]:>6} {v[2]:>6} {v[3]:>6}")
groups=defaultdict(list)
for _,cfg,fam,typ,k,src,inter,depth,temp,prior,ep,_,res in rows:
    if res: groups[(cfg,fam,typ,k,src,inter,depth,temp,prior,ep)].append(res)
print('\nconfiguration means')
final=[]
for key,results in sorted(groups.items()):
    cfg,fam,typ,k,src,inter,depth,temp,prior,ep=key
    maps=[x['mAP'] for x in results]; r1s=[x['R1'] for x in results]
    mm=statistics.mean(maps); rr=statistics.mean(r1s)
    ms=statistics.pstdev(maps) if len(maps)>1 else 0.0
    rs=statistics.pstdev(r1s) if len(r1s)>1 else 0.0
    print(f"{cfg:<30} {fam:<11} ep={ep:>3} n={len(results)} mAP={mm:.2f}+-{ms:.2f} R1={rr:.2f}+-{rs:.2f}")
    if ep==160: final.append((mm,rr,ms,rs,cfg,fam))
if final:
    baseline=next((x for x in final if x[4]=='l24_nolocal'),None)
    print('\nfinal deltas versus no-local')
    for mm,rr,ms,rs,cfg,fam in sorted(final,key=lambda x:(-x[0],-x[1])):
        dm=mm-baseline[0] if baseline else float('nan')
        dr=rr-baseline[1] if baseline else float('nan')
        print(f"{cfg:<30} {fam:<11} mAP={mm:.2f}+-{ms:.2f} ({dm:+.2f}) R1={rr:.2f}+-{rs:.2f} ({dr:+.2f})")
    best_map=max(final,key=lambda x:(x[0],x[1],-x[2]))
    best_r1=max(final,key=lambda x:(x[1],x[0],-x[3]))
    print(f"\nbest_mAP: {best_map[4]} family={best_map[5]} mean_mAP={best_map[0]:.2f} mean_R1={best_map[1]:.2f}")
    print(f"best_R1 : {best_r1[4]} family={best_r1[5]} mean_mAP={best_r1[0]:.2f} mean_R1={best_r1[1]:.2f}")
PY
}

echo "[L24] MODE=${MODE} experiments=${#EXPERIMENTS[@]} GPUs=${GPU_IDS} jobs=${MAX_JOBS}"
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then run_pool train; fi
if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then run_pool eval; fi
summarize | tee "${OUTPUT_BASE}/${SUMMARY_NAME}"
echo "[L24] Summary saved to ${OUTPUT_BASE}/${SUMMARY_NAME}"
