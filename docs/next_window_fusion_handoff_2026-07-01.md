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

---

# 2026-07-07 Window Update: Current Mainline and Negative Results

This section overrides several earlier assumptions in this document. The project has moved from plain Stage-FCU toward:

```text
Direction-aware Stage-FCU
  + same-scale map-Mamba fusion
  + MSEF refinement
  + triple descriptor inference
```

The most useful inference feature is:

```text
TEST.FEAT_MODE = 'mamba_fdmf_osnet'
```

This means the final test descriptor is:

```text
[MambaVision descriptor, map-Mamba/MSEF fused descriptor, OSNet descriptor]
```

The old `FDMF` name is kept for code/config compatibility, but the frequency decomposition path is no longer the important part. The effective current map fusion is:

```text
Mamba map + OSNet map
  -> align OSNet map to Mamba map size
  -> project OSNet map to Mamba channel dimension
  -> concat on channel
  -> 1x1 Conv / BN / SiLU
  -> Spatial Mamba
  -> MSEFBlock
  -> pooling
```

## Current Best Duke Result

The strongest Duke line so far is:

```text
Direction-aware Stage-FCU:
  stage2: OSNet -> Mamba
  stage3: Mamba -> OSNet

Map fusion:
  same-scale map-Mamba + MSEF

Inference:
  mamba_fdmf_osnet triple descriptor
```

Best reported Duke result:

```text
fdmf_msef_s2o2m_s3m2o
mAP: 84.0%
Rank-1: 91.5%
Rank-5: 96.0%
Rank-10: 97.1%
```

The same experiment's training-time fused feature alone was slightly lower:

```text
mAP: 83.9%
Rank-1: 91.4%
```

Interpretation:

- The triple descriptor at inference is better than relying only on the fused descriptor.
- The method works best when the branches remain partly independent instead of being over-merged.

## Pure Frequency Fusion Was Not Useful

The old FDMF frequency branch was tested and did not justify its cost.

Important Duke FDMF ablation results:

```text
fdmf_raw_no_mamba:
  mAP: 82.1
  Rank-1: 90.0

fdmf_mamba_fdmf:
  mAP: 83.4
  Rank-1: 90.8

pure no-frequency map-Mamba + MSEF:
  mAP: 83.7 / 83.8 range
  Rank-1: 91.2 / 91.6 range
```

Conclusion:

- The useful signal is not the frequency decomposition itself.
- The useful part is the same-scale map-Mamba refinement plus MSEF.
- The current mainline should not reintroduce heavy frequency filtering unless there is a very specific reason.

## MSEF Placement and Effect

MSEF from `multinex-main` is a single feature-map refinement block, not a two-branch fusion module.

Effective use in this project:

```text
concat maps
  -> 1x1 Conv
  -> SpatialMamba
  -> MSEFBlock
  -> pooling
```

Duke comparison:

```text
plain map-Mamba:
  mAP: 83.7
  Rank-1: 91.2

+ MSEF:
  mAP: 83.8
  Rank-1: 91.6

+ MSEF + residual scale:
  mAP: 83.8
  Rank-1: 91.6
```

Conclusion:

- MSEF is mildly useful, especially for Rank-1.
- Residual scale did not clearly improve over plain MSEF.
- Keep MSEF as a lightweight refinement, but do not overstate it as the main novelty.

## Stage-FCU With Map Fusion

After adding map-Mamba/MSEF, Stage-FCU was rerun.

Duke training summary:

```text
fdmf_msef_s2_o2m       mAP 83.5 / R1 90.9
fdmf_msef_s2_m2o       mAP 83.6 / R1 91.1
fdmf_msef_s2_bidir     mAP 83.4 / R1 90.9
fdmf_msef_s3_o2m       mAP 83.6 / R1 91.1
fdmf_msef_s3_m2o       mAP 83.7 / R1 91.1
fdmf_msef_s3_bidir     mAP 83.6 / R1 91.2
fdmf_msef_s23_bidir    mAP 83.7 / R1 90.8
fdmf_msef_s2o2m_s3m2o  mAP 83.9 / R1 91.4
```

Triple descriptor inference on the same weights:

```text
fdmf_msef_s2_o2m       mAP 83.6 / R1 90.8
fdmf_msef_s2_m2o       mAP 77.9 / R1 88.2
fdmf_msef_s2_bidir     mAP 83.2 / R1 90.5
fdmf_msef_s3_o2m       mAP 82.9 / R1 91.3
fdmf_msef_s3_m2o       mAP 83.9 / R1 91.2
fdmf_msef_s3_bidir     mAP 83.7 / R1 91.0
fdmf_msef_s23_bidir    mAP 83.8 / R1 91.5
fdmf_msef_s2o2m_s3m2o  mAP 84.0 / R1 91.5
```

Interpretation:

- Direction-aware Stage-FCU remains the best internal interaction setting.
- Stage 2 still prefers OSNet -> Mamba.
- Stage 3 still prefers Mamba -> OSNet.
- Some single-direction settings can look fine during training but behave poorly under triple descriptor inference.

## OSBBM Summary

OSBBM was fully implemented as an augmentation option, including grayscale rotation and random single-channel grayscale replacement.

For Duke with current best fusion:

```text
no_osbbm                    mAP 84.0 / R1 91.5
osbbm_p025_b8_m2_g05        mAP 84.0 / R1 91.6
osbbm_p050_b8_m2_g05        mAP 84.0 / R1 91.7
osbbm_p075_b8_m2_g05        mAP 83.6 / R1 91.6
osbbm_p050_b8_m1_g05        mAP 84.1 / R1 91.4
osbbm_p050_b8_m3_g05        mAP 83.1 / R1 91.2
osbbm_p050_b8_m4_g05        mAP 67.7 / R1 84.2
osbbm_p050_b6_m2_g05        mAP 82.9 / R1 90.0
osbbm_p050_b10_m2_g05       mAP 84.0 / R1 91.2
osbbm_p050_b12_m3_g05       mAP 83.8 / R1 90.9
osbbm_p050_b8_m2_g00        mAP 84.0 / R1 91.1
osbbm_p050_b8_m2_g025       mAP 84.1 / R1 91.3
osbbm_p050_b8_m2_g075       mAP 84.2 / R1 91.3
osbbm_p050_b8_m2_g10        mAP 84.1 / R1 91.7
osbbm_p025_b8_m4_g05        mAP 79.0 / R1 88.9
osbbm_p075_b8_m4_g05        mAP 58.9 / R1 78.6
```

Recommended OSBBM regularization setting:

```text
PROB: 0.5
NUM_BLOCKS: 8
NUM_MIX_BLOCKS: 2
GRAY_PROB: 0.75
MIXED_LABEL: False
SCHEDULE: always
```

Market/MSMT17 observation:

- OSBBM gives about +0.3 mAP on Market/MSMT17.
- Rank-1 may drop by a few tenths.
- Treat OSBBM as a regularization option, not a core method contribution.

OCC-Duke observation:

- Current network is not strong on OCC-Duke.
- OSNet may introduce too much occlusion noise.
- Do not make OCC-Duke the main dataset unless a visible-region/occlusion-aware mechanism is added.

## Stripe Mamba Ablation

A checkpoint was saved before stripe-Mamba work:

```text
commit a7e12a4
message: checkpoint before stripe mamba fusion
```

Stripe-Mamba support was then added:

```text
FDMF_STRIPE_DEPTH
FDMF_STRIPE_NUM
FDMF_STRIPE_SHARE_PARAMS
```

Market no-OSBBM ablation:

```text
fdmf_global_d1                 mAP 89.7 / R1 95.3
fdmf_global_d2                 mAP 89.3 / R1 95.2
fdmf_stripe_s1_g0_shared       last mAP 0.4 / R1 0.1, best mAP 85.7 / R1 93.8
fdmf_stripe_s1_g1_shared_k2    mAP 89.4 / R1 95.0
fdmf_stripe_s1_g1_shared       mAP 89.5 / R1 95.6
fdmf_stripe_s1_g1_shared_k8    mAP 89.4 / R1 95.2
fdmf_stripe_s1_g1_indep        mAP 89.7 / R1 95.4
fdmf_stripe_s2_g0_shared       mAP 89.7 / R1 95.4
```

Conclusion:

- Stripe-local scanning is not a core improvement.
- `global_d1` remains very strong.
- `global_d2` hurts.
- Stripe-only without global can collapse.
- `s1_g1_indep` and `s2_g0_shared` are only tiny Rank-1 variants, not convincing enough to replace the mainline.

## Negative Results From This Window

The following directions were attempted or clarified as ineffective:

| # | Direction | Specific idea | Location |
|---:|---|---|---|
| 1 | Stripe scan | Cut FDMF map into 4 stripes, scan each, concat | FDMF forward |
| 2 | SPD descriptor decomposition | Shared/private split plus orthogonality loss | after pooling + loss |
| 3 | Token concat fusion | Convert pooled mamba/osnet descriptors to tokens, Mamba scan, stitch back | after pooling |
| 4 | CBSF cross-branch modulation | OSNet global vector modulates MambaVisionMixer B and Delta parameters | inside Mamba scan |
| 5 | ETFFM token fusion | LN -> concat -> dual independent gates -> mutual gating -> projection | after pooling |
| 6 | USE upsample stripe | Upsample FDMF map 2x, cut stripes, GeM pool, add per-stripe inference/supervision | after FDMF + loss |
| 7 | S4-FCU | Add Stage-FCU at stage 4 before concat/residual injection | after backbone |

Important correction:

- Channel-Gated FCU had a script but was not actually run in the earlier count.
- The corrected count of poor directions is 7.

Overall interpretation:

```text
The system does not lack another interaction module.
It is more sensitive to over-interaction and branch homogenization.
```

The best behavior comes from:

- light direction-aware interaction inside stage 2/3,
- one fused map branch,
- keeping MambaVision, OSNet, and fused descriptors available at inference.

## Current Research Judgment

The strongest paper-positioning sentence is now:

```text
MambaVision and OSNet are complementary mainly as heterogeneous descriptor branches.
Light stage-level exchange and a small fused-map refinement branch help,
but overly strong stage/token/channel fusion weakens branch diversity and hurts ReID retrieval.
```

This explains both:

- why `mamba_fdmf_osnet` triple descriptor works;
- why many heavier fusion modules underperform.

## Recommended Next Steps From Here

1. Freeze the main structure:

```text
Direction-aware Stage-FCU
  stage2: OSNet -> Mamba
  stage3: Mamba -> OSNet
same-scale map-Mamba + MSEF
triple descriptor inference
```

2. Prefer low-cost inference/regularization studies over new heavy modules:

```text
branch-wise L2 normalize + weighted concat
Mamba : FDMF : OSNet weight search
DropBranch / DropDescriptor during training
OSBBM as optional regularization
```

3. If trying one more map-fusion variant, the only promising one is late compression:

```text
Mamba map, OSNet map
  -> cat to 2C
  -> Mamba in 2C token space
  -> reduce to C
  -> MSEF
  -> pooling
```

Rationale:

- Current FDMF compresses two maps to C before Mamba.
- Similar Mamba fusion papers usually scan spatial tokens, not channel dimension as the main sequence.
- Channel modeling is better used as gate/attention/auxiliary Channel-SSM, not the primary scan axis.

Do not prioritize:

- more stripe-local scanning,
- pooled-token Mamba,
- invasive Mamba internal modulation,
- stronger orthogonality loss,
- deeper global Mamba.

## Current Working Tree Notes

At the time of this update, relevant local status included:

```text
M config/defaults.py
M configs/Market/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml
M model/make_model.py
M processor/processor.py
?? scripts/run_duke_osnet_fdmf_stripe_mamba.sh
?? scripts/run_market_osnet_fdmf_stripe_mamba.sh
?? data/
?? datasets/Occluded_Duke/
?? etffm_backup/
?? s4fcu_backup/
?? tmp/
?? use_backup/
```

Do not delete or revert these unless explicitly asked. Data and backup folders are intentionally untracked.
