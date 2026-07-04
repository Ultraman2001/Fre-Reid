# Fusion Handoff: MambaVision + OSNet on DukeMTMC-reID

Date: 2026-07-01  
Repo: `Fre-Reid`  
Main task: continue research on MambaVision + OSNet dual-branch fusion for DukeMTMC-reID.

## Read This First

The current mainline should be treated as:

- Keep **descriptor concat** and **Stage-FCU internal feature interaction**.
- ETFFM / token-fusion code has been removed from the current mainline.
- CTGMF code had also been removed earlier and should not be assumed present.
- The best confirmed direction so far is **Stage-FCU**, especially asymmetric stage-level exchange.

Current important files:

- `model/make_model.py`
- `config/defaults.py`
- `configs/DukeMTMC/mambavision_tiny_osnet_stage_fcu_b64k4.yml`
- `scripts/run_duke_osnet_stage_fcu.sh`
- `loss/make_loss.py`
- `solver/make_optimizer.py`

Important server pretrained paths:

- MambaVision: `/workspace/pretrained/mambavision_tiny_1k.pth.tar`
- OSNet: `/workspace/pretrained/osnet_x1_0_imagenet.pth`

## Current Code State

`MODEL.OSNET_FUSION.FUSION_TYPE` currently supports only:

- `descriptor`
- `stage_fcu`

Removed from mainline:

- `token_fusion`
- `MOETFFM`
- `TOKEN_FUSION_*`
- `MAMBA_LOSS_WEIGHT`
- `scripts/run_duke_osnet_token_fusion.sh`
- `configs/DukeMTMC/mambavision_tiny_osnet_token_fusion_b64k4.yml`

Stage-FCU was kept. It now also supports per-stage direction override:

- `MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION`
- `MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION`

If these are empty, the model falls back to `FCU_DIRECTION`.

Current Stage-FCU script:

```bash
bash scripts/run_duke_osnet_stage_fcu.sh 0,1 2
bash scripts/run_duke_osnet_stage_fcu.sh summary ./logs/Duke/osnet_stage_fcu_directional
```

The script default output directory is:

```bash
./logs/Duke/osnet_stage_fcu_directional
```

It skips existing `train_log.txt` unless:

```bash
FORCE_RUN=1 bash scripts/run_duke_osnet_stage_fcu.sh 0,1 2
```

## Confirmed Results

### Stage-FCU Single-Direction Ablation

These were completed to 160 epochs.

| Experiment | Stages | Direction | mAP | Rank-1 | Notes |
|---|---:|---|---:|---:|---|
| `stage_fcu_s2_o2m_osw1_fuw1` | `[2]` | OSNet -> Mamba | 82.9 | 91.0 | strong |
| `stage_fcu_s2_m2o_osw1_fuw1` | `[2]` | Mamba -> OSNet | 82.7 | 90.7 | close |
| `stage_fcu_s3_o2m_osw1_fuw1` | `[3]` | OSNet -> Mamba | 81.8 last / 81.9 best | 90.1 last / 90.6 best | weaker |
| `stage_fcu_s3_m2o_osw1_fuw1` | `[3]` | Mamba -> OSNet | 82.9 | 90.9 | strong |

Interpretation:

- Stage 2 prefers **OSNet -> Mamba**.
- Stage 3 prefers **Mamba -> OSNet**.
- This motivated the mixed-direction experiment.

### Mixed Stage-FCU Direction

Best currently reported result:

```text
stage2: OSNet -> Mamba
stage3: Mamba -> OSNet

Epoch 160:
mAP: 83.2%
Rank-1: 90.8%
Rank-5: 95.5%
Rank-10: 96.9%
```

This is the most promising confirmed fusion result so far.

Script experiment name:

```text
stage_fcu_s2o2m_s3m2o_osw1_fuw1
```

Script spec:

```bash
"stage_fcu_s2o2m_s3m2o_osw1_fuw1|[2,3]|s2_o2m_s3_m2o|0.1|1.0|1.0"
```

Implementation detail:

- The script maps `s2_o2m_s3_m2o` to:
  - global `FCU_DIRECTION='bidirectional'`
  - `FCU_STAGE2_DIRECTION='osnet_to_mamba'`
  - `FCU_STAGE3_DIRECTION='mamba_to_osnet'`

## Fusion Methods Tried

### 1. Raw Descriptor Concat

Basic late descriptor fusion:

```text
fused = concat(mamba_feat, osnet_feat)
```

This is the stable baseline. In the token-fusion baseline run, descriptor concat reached:

```text
descriptor_baseline_mw05_osw02_fuw1
mAP: 82.1
Rank-1: 90.8
```

Do not over-interpret the exact value because it was run inside a token-fusion ablation setup, but it is useful as a stability reference.

### 2. Stage-FCU Internal Feature Interaction

This is currently the best path.

Concept:

- Exchange feature maps inside the two branches before final pooling.
- It is not just late descriptor concat.
- It allows one branch to inject spatial/channel information into the other branch at stage 2 or stage 3.

Directions:

- `osnet_to_mamba`
- `mamba_to_osnet`
- `bidirectional`

Current useful setting:

```text
stage2: OSNet -> Mamba
stage3: Mamba -> OSNet
init_scale: 0.1
osnet_loss_weight: 1.0
fused_loss_weight: 1.0
```

Why it seems reasonable:

- Stage 2 has more mid-level texture/local cues; OSNet helps Mamba.
- Stage 3 has stronger semantic/global cues; Mamba helps OSNet.

### 3. CTGMF

CTGMF was an attempted fusion design around token/global-map complementary interaction.

Experiment groups discussed:

- `descriptor_baseline_osw1_fuw1`
- `ctgmf_full_tw05_pw05_osw1_fuw1`
- `ctgmf_no_token_agree_tw05_pw05`
- `ctgmf_no_token_guidance_tw05_pw05`
- `ctgmf_no_patch_mamba_tw05_pw05`
- `ctgmf_no_aux_osw1_fuw1`

Important result:

- `ctgmf_full_tw05_pw05_osw1_fuw1` was already behind the descriptor baseline by about **6.6 mAP at 40 epochs**.

Conclusion:

- CTGMF was not promising in this setup.
- It complicated the branch supervision and did not preserve a strong identity-feature path.
- It was removed from the current mainline.

### 4. TE-TransReID-Inspired ETFFM Token Fusion

We analyzed TE-TransReID Section 3.4 and tried to mimic its token-level fusion idea.

TE-TransReID idea:

```text
Normalize two token vectors.
Generate mutual gates from concatenated token features.
Compute bidirectional gated features:
  f_i,j
  f_j,i
Combine them with original concat feature:
  f_token = Mutual-Gated(f_i,j, f_j,i, f_c)
```

Important clarification from the discussion:

- In the TE-TransReID-style logic, the original concat feature `f_c` itself acts like a residual/identity-preserving path.
- The gated features should not replace the raw concat path.
- The gated features are lower-dimensional token interactions, while `f_c` carries original branch information.

What happened in our implementation:

- The first implementation was too simplified and unstable.
- The model evaluated only the bad fused/token branch by default via `TEST.FEAT_MODE='concat'`.
- At epoch 20, the token-fusion branch showed:

```text
TOKEN_FUSION[ID=6.55, Tri=0.6931]
Validation:
mAP: 0.1%
Rank-1: 0.0%
```

Interpretation:

- `ID=6.55` is basically `log(num_classes)`, meaning the fused classifier learned almost no identity information.
- `Tri=0.6931` is the soft-triplet collapse value.
- Mamba and OSNet branches themselves were not necessarily broken; the evaluated fused feature was broken.

Conclusion:

- ETFFM/token-fusion was removed from current mainline.
- Do not continue from that code unless intentionally rebuilding it from scratch with a much stronger identity-preserving path and separate branch evaluation.

## Important Pitfall: L2 Normalization Before Product

We discussed a potential trap:

```text
m = L2(mamba_feat)
o = L2(osnet_feat)
then use |m-o| and m*o
```

Concern:

- A 512-D L2-normalized vector has average magnitude around `1 / sqrt(512) = 0.044`.
- Hadamard products can fall near `1e-3`.
- This can make interaction features weak, especially if followed by randomly initialized MLPs.

Correction:

- This is not automatically fatal in all networks, because MLP weights and normalization can rescale.
- But for a newly inserted fusion branch with weak supervision, it can easily create a poor learning signal.
- Prefer using BN/LN and raw or lightly normalized features with a residual/concat identity path.

## Loss and Evaluation Notes

OSNet fusion loss currently expects three branches:

```text
score = [mamba_score, osnet_score, fused_score]
feat  = [mamba_feat, osnet_feat, fused_feat]
```

Weights:

```text
mamba weight = 1.0
osnet weight = MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT
fused weight = MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT
```

For Stage-FCU experiments we have mainly used:

```text
OSNET_LOSS_WEIGHT = 1.0
FUSED_LOSS_WEIGHT = 1.0
```

Evaluation:

- `processor._select_eval_feature` uses `TEST.FEAT_MODE`.
- Default useful mode for fusion is usually `concat`.
- In descriptor/stage-FCU mode, `concat` means the fused descriptor from `concat(mamba_feat, osnet_feat)`.
- In the removed token-fusion mode, `concat` was misleading because it selected the broken token-fusion output.

## Current Script Behavior

`scripts/run_duke_osnet_stage_fcu.sh` currently runs five Stage-FCU experiments:

```text
stage_fcu_s2_o2m_osw1_fuw1
stage_fcu_s2_m2o_osw1_fuw1
stage_fcu_s3_o2m_osw1_fuw1
stage_fcu_s3_m2o_osw1_fuw1
stage_fcu_s2o2m_s3m2o_osw1_fuw1
```

Summary-only:

```bash
bash scripts/run_duke_osnet_stage_fcu.sh summary ./logs/Duke/osnet_stage_fcu_directional
```

Normal run:

```bash
bash scripts/run_duke_osnet_stage_fcu.sh 0,1 2
```

Force rerun:

```bash
FORCE_RUN=1 bash scripts/run_duke_osnet_stage_fcu.sh 0,1 2
```

## Related External Ideas Discussed

### TE-TransReID

Useful idea:

- Separate token-level fusion and feature-map-level fusion.
- Token fusion uses mutual gates and raw concat feature.
- Feature-map fusion is handled separately.

Risk in our setting:

- TE-TransReID has no public code.
- Directly reproducing paper equations can be ambiguous.
- Our first token-fusion attempt failed badly.

Takeaway:

- Do not resume ETFFM immediately.
- If revisited, start with descriptor concat as explicit preserved output and evaluate all branches separately.

### MambaMIC FMIAM

We discussed it conceptually as a cross-branch interaction style:

- More directly relevant to feature-map cross-branch fusion than ETFFM.
- Worth revisiting when designing the next map-level module.

### Multinex MSEF

MSEF in `multinex-main` is not a two-branch fusion module. It is a single feature-map self-enhancement block:

```text
x_norm = LayerNorm(x)
x1 = DWConv3x3(x_norm)
x2 = SE(x_norm)
out = x + x1 * x2
```

Potential use:

- Do not use it for pooled token fusion.
- It may be useful as a lightweight **post-fusion feature-map refiner** after Mamba/OSNet maps are aligned and fused.

Suggested safe ReID adaptation:

```text
out = x + gamma * (DWConv(LN(x)) * SE(LN(x)))
```

with small `gamma`, e.g. `0.1` or `1e-3`, because original MSEF uses `tanh` SE and may disturb identity features if inserted aggressively.

## Recommended Next Steps

1. Treat Stage-FCU mixed-direction as the current main result.

   Priority:

   ```text
   stage2: OSNet -> Mamba
   stage3: Mamba -> OSNet
   ```

2. Run or confirm repeated seed if time allows.

   The 83.2 mAP result is promising, but a repeated seed would help verify stability.

3. If proposing a new module, focus on feature-map fusion, not token fusion first.

   The next sane direction:

   ```text
   Stage-FCU / aligned maps
      -> lightweight map refinement
      -> pooling
      -> descriptor concat/fused loss
   ```

4. Candidate map module:

   Use an MSEF-like or FMIAM-like block after map alignment, with:

   - residual identity path
   - small learnable scale
   - separate ablation for branch map refinement vs fused map refinement

5. Avoid repeating CTGMF or ETFFM immediately.

   They consumed time and underperformed:

   - CTGMF lagged baseline by ~6.6 mAP at epoch 40.
   - ETFFM/token fusion collapsed to 0.1 mAP at epoch 20.

## Caution About Git State

The working tree has unrelated deleted scripts and untracked data folders. Do not revert them unless explicitly asked.

Known current relevant modified files:

- `config/defaults.py`
- `model/make_model.py`
- `scripts/run_duke_osnet_stage_fcu.sh`

`loss/make_loss.py` and `solver/make_optimizer.py` may show line-ending related status in some environments, but after ETFFM removal they should have no meaningful token-fusion content.

Before further changes, run:

```bash
rg -n "MOETFFM|token_fusion|TOKEN_FUSION|ETFFM|MAMBA_LOSS_WEIGHT" model config loss solver scripts configs
```

Expected result:

```text
no matches
```

## One-Sentence Research Positioning

The strongest current contribution candidate is not late token fusion, but **direction-aware stage-level cross-branch feature exchange**, where OSNet injects mid-level local cues into Mamba at stage 2 and Mamba injects stronger semantic/global cues into OSNet at stage 3.
