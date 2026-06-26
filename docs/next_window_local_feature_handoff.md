# Next Window Handoff: Local/Fine-Grained Feature Work

This document summarizes the current Fre-ReID experimental state before opening a new context window. The next window should use this as the starting point and should avoid repeating the failed local/fine-grained feature routes listed below.

## Latest Update Before Next Window (2026-06-26)

Current tracked direction is back to the reliable baseline family:

- MambaVision tiny + PADE-style PAM.
- `PAM_AUGMENTED_LOSS_WEIGHT = 0.5`, giving effective BA/CA/EA weights of 0.50/0.25/0.25.
- Scheduled OSBBM on BA only: `prob=0.25`, `num_blocks=8`, `num_mix_blocks=2`, `cycle`, `start=21`, `end=120`, `period=20`, `on=10`.

Recent experimental branches have been reverted from the code:

- R-PVTR was implemented and then reverted by `4ba29d7`.
- HULM was implemented in `6799b7e` and then reverted by `0240e4c`.
- A checkpoint before HULM revert exists at `e17dd07`.
- The current user-accepted `mamba_vision_reid.py` state removes FSLoRA spatial-context helper logic. Treat FSLoRA as out of scope for the next window unless explicitly requested. The current PAM + schedule-OSBBM configs do not enable FSLoRA.

Important latest results:

```text
R-PVTR final:
Epoch 160 mAP 59.3, Rank-1 67.7
Monitor: gamma=-0.488043, score_mean=0.4177, score_std=0.1860, tau=0.0584

HULM stage2_pre final:
Epoch 160 mAP 58.4, Rank-1 67.0
Monitor: beta=0.110954, global_norm=30.514, local_norm=36.693

HULM stage4 final:
Epoch 160 mAP 58.6, Rank-1 65.9
Monitor: beta=0.104012, global_norm=30.287, local_norm=31.479
```

Interpretation:

- R-PVTR learned a negative residual coefficient, effectively suppressing features instead of improving occlusion robustness.
- HULM learned non-trivial beta around 0.10, and the local branch norm was comparable to or larger than the global branch norm. This injected a strong local perturbation that hurt validation performance.
- Stage2 pre-downsample local features did not fix the issue. Higher spatial resolution alone did not produce better local identity cues.
- Do not continue R-PVTR/HULM without a substantially different mechanism. If a final rescue is attempted, cap residual strength much lower (`beta_max <= 0.05`) and use it only as a sanity check, not as the main next direction.

Recommended next focus:

- Keep the code simple and return to the strong PAM + scheduled OSBBM baseline.
- Prefer OSBBM-aware constraints or branch-consistency losses over new local descriptor branches.
- A promising low-risk next idea is BA/CA/EA feature consistency or OSBBM mask/visible-token supervision, because these build on the augmentation that is already helping instead of adding another independent local branch.

## Main Goal Going Forward

Continue improving occluded person re-identification on OCC-Duke by strengthening local or fine-grained features on top of the current MambaVision tiny + PADE-style PAM baseline.

The most reliable current direction is:

- Keep PADE-style PAM.
- Use PADE-style loss weighting: BA = 0.5, CA = 0.25, EA = 0.25.
- Use OSBBM only as a scheduled mid-training robustness perturbation, not as an always-on augmentation.
- New work should focus on local/fine-grained feature modeling that adds useful information beyond BA/CA/EA instead of simply adding weak auxiliary stripe heads.

## Current Best Practical Baseline

Recommended baseline config:

- Config base: `configs/OCC_Duke/mambavision_tiny_transreid_pam_padeaug_b64k4.yml`
- Backbone: `mambavision_tiny_TransReID`
- Dataset: `occ_duke`
- Batch: `64`, K = `4`
- Loss: softmax + triplet, label smoothing on, soft triplet (`NO_MARGIN: True`)
- Pooling: `gem`
- PAM enabled with PADE-style augmentation
- PAM branch loss weight: `SOLVER.PAM_AUGMENTED_LOSS_WEIGHT = 0.5`

Important PADE-style loss behavior:

```text
ID_LOSS  = (BA + 0.5 * (CA + EA)) / 2
TRI_LOSS = (BA + 0.5 * (CA + EA)) / 2
```

Effective branch weights:

```text
BA = 0.50
CA = 0.25
EA = 0.25
```

This matches the PADE-main style more closely than equal BA/CA/EA weighting.

## Best OSBBM Schedule So Far

Best current scheduled OSBBM setting:

```yaml
INPUT:
  OSBBM:
    ENABLED: True
    PROB: 0.25
    NUM_BLOCKS: 8
    NUM_MIX_BLOCKS: 2
    GRAY_PROB: 0.50
    APPLY_TO: 'base'
    SCHEDULE: 'cycle'
    START_EPOCH: 21
    END_EPOCH: 120
    PERIOD_EPOCHS: 20
    ON_EPOCHS: 10

SOLVER:
  PAM_AUGMENTED_LOSS_WEIGHT: 0.5
```

Interpretation:

- Epochs 1-20: no OSBBM, let BA/PAM representation stabilize.
- Epochs 21-120: cycle OSBBM, 10 epochs on and 10 epochs off.
- Epochs 121-160: no OSBBM, let the model converge back toward clean BA input.

This supports the idea that OSBBM should be treated as a curriculum-style robustness perturbation. It should not permanently destroy BA, because BA is the complete-input anchor in PADE/PAM.

## Key Result Tables

### Base + PAM(w=0.5), No OSBBM

After adjusting PADE-style PAM loss weight:

```text
Epoch 160:
mAP 59.4
Rank-1 68.3
Rank-5 81.1
Rank-10 85.4
```

This setup still has training oscillation, but it is already stronger than several earlier local-branch attempts.

### PAM(w=0.5) + OSBBM Schedule Ablation

```text
name                             last_ep  last_mAP  last_R1  best_ep  best_mAP  best_R1
pamw05_osbbm_m2_warm_cycle_21_120 160     59.5      68.2     140      59.8      68.4
pamw05_osbbm_m2_mid_41_120        160     59.2      67.1     120      59.7      68.8
```

Conclusion:

- `warm_cycle_21_120` is more stable.
- `mid_41_120` can peak at higher Rank-1 but drops too much by the final epoch.

### Switch-Length Ablation

Script: `scripts/run_pamw05_osbbm_switch_ablation.sh`

```text
name                             start  end   on   off  last_ep  last_mAP last_R1  best_ep  best_mAP best_R1
pamw05_m2_c21_120_p20_on10       21     120   10   10   160      60.0     68.9     120      60.0     69.3
pamw05_m2_c21_120_p20_on5        21     120   5    15   160      59.5     67.3     120      59.6     68.5
pamw05_m2_c21_120_p20_on15       21     120   15   5    160      59.5     67.8     160      59.5     67.8
pamw05_m2_c21_120_p10_on5        21     120   5    5    160      59.5     67.3     160      59.5     67.3
pamw05_m2_c21_120_p30_on10       21     120   10   20   160      59.6     68.3     120      59.6     69.0
pamw05_m2_c21_120_p40_on20       21     120   20   20   160      59.0     67.5     140      59.0     68.1
pamw05_m2_c41_120_p20_on10       41     120   10   10   160      59.5     68.2     160      59.5     68.2
pamw05_m2_c21_140_p20_on10       21     140   10   10   160      59.7     67.9     160      59.7     67.9
```

Conclusion:

- Best setting is `start=21, end=120, period=20, on=10`.
- `on=5` is too weak.
- `on=15` is too strong.
- `period=10` switches too frequently.
- `period=30/40` weakens the effect or destabilizes training.
- `start=41` is too late.
- `end=140` keeps perturbation too late into convergence.

## Earlier OSBBM Ablation Before PAM Loss Weight Fix

These were run before changing PAM loss weight to the PADE-style `0.5` setting:

```text
name                           sched    start  end   last_ep  last_mAP last_R1  best_ep  best_mAP best_R1
osbbm_m2_always                always   1      0     160      58.7     66.5     120      58.8     67.4
osbbm_m2_cycle_10on10off       cycle    1      0     160      59.4     67.9     160      59.4     67.9
osbbm_m2_warm_cycle_21_160     cycle    21     160   160      59.6     67.5     160      59.6     67.5
osbbm_m2_warm_cycle_21_120     cycle    21     120   160      59.3     68.0     160      59.3     68.0
osbbm_m2_mid_41_120            range    41     120   160      59.6     67.7     120      59.7     68.8
```

Useful lesson:

- Always-on OSBBM is bad.
- Mid-training or cyclic OSBBM is better.
- Scheduled OSBBM improves robustness only when BA still has enough clean-input training.

## PADE-main Compatibility Notes

PADE-main was inspected locally. Its PAM-style setup is approximately:

- BA image: clean resized image, normalized.
- CA image: crop-augmented image with padding and random resized crop.
- EA image: random-erased image.
- Shared backbone for BA/CA/EA.
- Separate BN/classifier heads for BA/CA/EA.
- Loss weighting with label smoothing effectively gives BA half of the loss and CA/EA the other half.

Current Fre-ReID implementation is not literally identical to PADE-main because the backbone is MambaVision and the dataloader/transform pipeline may differ, but the current `AUG_MODE: 'pade'` + `PAM_AUGMENTED_LOSS_WEIGHT: 0.5` is the closest current PADE-style setting.

## Local/Fine-Grained Feature Routes Already Tried

The following directions were tried and are not worth repeating without a substantially different design.

### 1. BA Local Stripe Auxiliary Branch

Idea:

- Add local stripe heads on BA feature maps.
- Use stripe-level ID/triplet supervision.

Observed issue:

- Auxiliary local branches were too weak compared with the global backbone branch.
- They added loss terms but did not provide enough independent representation power.
- Gains were not competitive with simply improving PAM/OSBBM scheduling.

Current status:

- Do not repeat a simple lightweight stripe-head-only design.
- If local stripes are revisited, they need stronger feature extraction, alignment, or relation modeling rather than only classification heads.

### 2. Stage3 Local Branch

Idea:

- Take Stage3 features and add local stripe supervision.
- Explore number of stripes, loss weight, grad scale, and OSBBM combinations.

Representative results:

```text
name                         s    w      smr    gs    osbbm   last_mAP last_R1  best_mAP best_R1
s3_smr_s2_w0025              2    0.025  True   0.20  False   59.3     67.8     59.3     67.8
s3_smr_s2_w005               2    0.05   True   0.20  False   59.0     67.6     59.1     67.9
s3_smr_s2_w005_gs01          2    0.05   True   0.10  False   59.2     67.5     59.4     67.9
s3_smr_s2_w005_gs05          2    0.05   True   0.50  False   59.3     68.1     59.3     68.1
s3_smr_s4_w005               4    0.05   True   0.20  False   58.4     67.5     58.4     67.5
s3_plain_s2_w005             2    0.05   False  0.20  False   59.1     66.8     59.1     66.8
s3_smr_s2_w005_osbbm_m2      2    0.05   True   0.20  True    59.8     68.1     59.8     68.1
s3_smr_s4_w005_osbbm_m2      4    0.05   True   0.20  True    59.4     68.1     59.4     68.1
s3_smr_s2_w005_osbbm_m4      2    0.05   True   0.20  True    58.8     67.1     58.8     67.1
```

Conclusion:

- Stage3 local supervision alone is not enough.
- Best-looking improvements were mostly from OSBBM m2 rather than the Stage3 local branch itself.
- More stripes often hurt.
- This route should be dropped unless the Stage3 branch is redesigned as a real local feature processor, not a shallow auxiliary loss.

### 3. Stage4 Local Branch

Idea:

- After discussing that Stage3 local features may be too shallow, tried Stage4 local processing.
- Considered stronger local feature branch closer to global feature depth.
- Stage4 local used BA/CA/EA local stripe variants and local losses.

Representative results:

```text
name                           s    w       tri    tw      gs    osbbm   last_mAP last_R1  best_mAP best_R1
s4local_s2_w0025_gs02          2    0.025   False  0.000   0.20  False   59.0     67.3     59.0     67.3
s4local_s2_w005_gs02           2    0.050   False  0.000   0.20  False   59.0     67.3     59.0     67.3
s4local_s2_w010_gs02           2    0.100   False  0.000   0.20  False   58.9     67.1     58.9     67.1
s4local_s2_w005_gs05           2    0.050   False  0.000   0.50  False   59.0     67.0     59.0     67.0
s4local_s4_w005_gs02           4    0.050   False  0.000   0.20  False   58.9     67.7     58.9     67.5
s4local_s2_w005_tri0025        2    0.050   True   0.025   0.20  False   59.1     66.9     59.1     66.9
s4local_s2_w005_osbbm_m2       2    0.050   False  0.000   0.20  True    59.0     67.2     59.0     67.2
```

Conclusion:

- Effect is small and not worth keeping.
- Local branch still did not extract enough independent local/fine-grained information.
- Adding triplet did not fix it.
- OSBBM did not rescue it.

### 4. Replacing EA With OSBBM

Idea:

- Replace PADE EA branch with OSBBM/OBSSM-style image.
- Test whether structured occlusion is a better augmentation than random erasing.

Results:

```text
name                   prob  blocks  mix  last_mAP last_R1  best_mAP best_R1
replace_ea_p015_b8_m2  0.15  8       2    57.9     67.3     57.9     67.3
replace_ea_p020_b8_m2  0.20  8       2    58.4     67.4     58.4     67.4
replace_ea_p025_b8_m2  0.25  8       2    57.6     66.6     57.6     66.6
replace_ea_p030_b8_m2  0.30  8       2    58.2     66.7     58.2     66.7
replace_ea_p025_b8_m3  0.25  8       3    57.5     66.5     57.5     66.5
replace_ea_p025_b8_m4  0.25  8       4    58.4     66.4     58.4     66.4
```

Conclusion:

- Replacing EA is clearly worse than original PADE-style EA.
- EA should remain random-erasing style.
- OSBBM should be applied to BA only as scheduled extra perturbation, not as an EA replacement.

### 5. OSBBM Always-On or Applied to All PAM Branches

Earlier OSBBM ablations:

```text
name                       prob   mix   apply  last_mAP last_R1  best_mAP best_R1
osbbm_p015_b8_m4_base      0.15   4     base   59.3     66.9     59.3     66.9
osbbm_p025_b8_m4_base      0.25   4     base   58.7     65.6     58.7     65.8
osbbm_p050_b8_m4_base      0.50   4     base   58.5     66.1     58.5     66.1
osbbm_p025_b8_m2_base      0.25   2     base   59.5     67.0     59.8     68.7
osbbm_p025_b8_m6_base      0.25   6     base   58.9     66.1     58.9     66.9
osbbm_p025_b8_m4_all       0.25   4     all    59.8     67.0     59.8     67.0
```

Conclusion:

- `m2` is better than heavier mix settings.
- Applying to all branches is not convincingly better and may interfere with the PADE structure.
- Keep `APPLY_TO: base`.
- Avoid high probability or heavy mix blocks.

### 6. R-PVTR / DPEFormer-Inspired Proxy-Variance Token Recalibration

Idea:

- Use a learnable proxy token plus variance gating to score final MambaVision tokens.
- Apply a residual token recalibration before pooling.
- This was inspired by DPEFormer/TSA-style token scoring.

Result:

```text
Epoch 160:
mAP 59.3
Rank-1 67.7
gamma_mean -0.488043
score_mean 0.4177
tau 0.0584
```

Conclusion:

- Worse than the PAM + scheduled OSBBM baseline.
- The learned residual coefficient became negative, so the module learned to suppress rather than enhance token features.
- Do not repeat proxy+variance token scoring in this form.
- If revisiting token scoring, tie it directly to OSBBM masks or visible-region supervision instead of unsupervised proxy learning.

### 7. HULM: High-Resolution Upsampled Local Mamba Branch

Idea:

- Keep the stage4 global GeM descriptor as the main path.
- Add a local branch from either stage4 upsampled to 32x8 or stage2 pre-downsample 32x16.
- Split the local map into 2 parts and add it back through a small non-negative residual coefficient.

Results:

```text
HULM stage2_pre:
mAP 58.4
Rank-1 67.0
beta 0.110954
global_norm 30.514
local_norm 36.693

HULM stage4:
mAP 58.6
Rank-1 65.9
beta 0.104012
global_norm 30.287
local_norm 31.479
```

Conclusion:

- Clearly worse than baseline.
- Local branch features became strong enough to disturb the global descriptor but did not generalize.
- Stage2 pre-downsample 32x16 features did not help.
- Do not continue height-upsample local branch or stage2-pre local descriptor in this form.

## Current Code/Script Map

Important implementation files:

- `config/defaults.py`
  - Contains OSBBM config keys and schedule fields.
- `processor/processor.py`
  - Applies OSBBM during training.
  - Contains epoch schedule logic for `always`, `range`, and `cycle`.
- `utils/osbbm.py`
  - Implements OSBBM batch augmentation.
- `loss/make_loss.py`
  - Implements PADE/PAM branch loss weighting.
  - Current important setting is `PAM_AUGMENTED_LOSS_WEIGHT = 0.5`.
- `model/make_model.py`
  - PAM multi-branch forward/head logic.
- `model/backbones/mambavision/mamba_vision_reid.py`
  - MambaVision backbone.

Useful scripts:

- `scripts/run_pamw05_osbbm_switch_ablation.sh`
  - Best OSBBM schedule ablation script.
- `scripts/run_pamw05_osbbm_schedule_ablation.sh`
  - Smaller schedule ablation script.
- `scripts/run_osbbm_schedule_ablation.sh`
  - Older schedule script before PAM weight cleanup.
- `scripts/run_osbbm_ablation.sh`
  - Basic OSBBM probability/mix/apply-to ablation.
- `scripts/run_stripeaux_ablation.sh`
  - Earlier stripe auxiliary experiments.
- `scripts/run_stripectx_ablation.sh`
  - Earlier stripe context experiments.
- `scripts/run_localjpm_ablation.sh`
  - Earlier local/JPM-style direction.

## What To Avoid Next

Avoid spending more time on:

- Simple local stripe classifier heads.
- Stage3 shallow auxiliary stripe loss.
- Stage4 local stripe auxiliary loss in its previous form.
- R-PVTR / proxy-variance token recalibration in its previous form.
- HULM stage4 or stage2-pre local residual branch.
- Replacing EA with OSBBM.
- Always-on OSBBM.
- OSBBM applied to all PAM branches.
- Increasing OSBBM strength blindly.
- Late OSBBM ending after epoch 120.
- Treating best-epoch-only improvements as strong evidence when final epoch drops heavily.

## Promising Next Directions

The next work should be a stronger local/fine-grained feature design. Good candidates:

### Direction A: Strong Local Branch With Shared Early Backbone

Use shared MambaVision stages for BA/CA/EA, then add a real local branch that has enough capacity.

Possible design:

- Shared global backbone up to a late stage.
- Local branch uses the same high-level feature map.
- Split vertical stripes or windows.
- Each local region goes through a lightweight but nontrivial local processor, not only pooling + classifier.
- Fuse local descriptors with global descriptor at inference.

Important:

- Local branch should improve inference features, not only provide training auxiliary loss.
- If local features are not included in the final descriptor, the branch may not help enough.

### Direction B: Part-Aware Token/Stripe Interaction

Instead of independent stripe heads:

- Split feature map into horizontal body stripes.
- Add cross-stripe relation modeling.
- Use attention/Mamba scanning over stripe tokens.
- Let visible body parts communicate with adjacent stripes.

This may be better for occlusion than isolated stripe classifiers.

### Direction C: Occlusion-Aware Local Gating

Use local confidence/gating:

- Estimate stripe visibility or reliability.
- Downweight likely occluded stripes.
- Fuse visible stripes more strongly.
- Keep global BA feature as fallback.

This aligns better with OCC-Duke than simply forcing every stripe to classify identity.

### Direction D: Descriptor-Level Global + Local Fusion

At inference, use:

```text
final_feature = concat(global_feature, local_feature_1, ..., local_feature_k)
```

or a learned projection/fusion:

```text
final_feature = projection([global, weighted_local])
```

Previous local branches were too auxiliary-loss-oriented. The next design should make local descriptors part of the final retrieval representation.

### Direction E: Stabilization Strategies For PAM

PAM still creates epoch-to-epoch oscillation. Possible low-risk stabilization experiments:

- EMA evaluation if code path is already available.
- Best-checkpoint selection by mAP instead of only final checkpoint.
- Slightly lower drop path or LR for PAM runs.
- Keep OSBBM schedule `21-120, 10 on / 10 off` as the current default when testing new local modules.

Do not start with many local-module hyperparameters. First verify one clean design against:

1. Base + PAM(w=0.5), no OSBBM.
2. Base + PAM(w=0.5) + OSBBM schedule `21-120, p20/on10`.

## Recommended First Experiment For Next Window

Use the current best training strategy as the baseline and only change the local module:

```text
Baseline:
PADE-style PAM(w=0.5)
OSBBM m2 scheduled from epoch 21 to 120
20-epoch period, 10 on / 10 off
APPLY_TO=base
```

Then test a local branch that contributes to the final descriptor, not merely auxiliary loss.

Minimum comparison table should include:

```text
name
local_module
uses_local_in_inference
osbbm_schedule
last_mAP
last_R1
best_epoch
best_mAP
best_R1
final_minus_best_gap
```

The final-minus-best gap is important because many configurations peak early but degrade by epoch 160.

---

## Additional Failed Routes (2026-06-20/21 window)

### 6. BPBreID (Body Part Attention + PifPaf Masks)

Idea: PixelToPartClassifier on feat_s4 → K+1 attention maps → GWAP part features → GiLt loss.

Why failed: MambaVision's 16×8 feature map is too coarse for precise part boundary learning. Part attention maps at 16×8 cannot resolve body part boundaries. Observed: Standard Duke 79.5% vs baseline 81.9%, OCC-Duke also degraded.

### 7. VBSM (Vertical Body Stripe Mamba)

Idea: Width-pool feat_s4 into stripe prototypes → bidirectional MambaVisionMixer scan over body axis → broadcast context back.

Why failed: Width pooling mixes occluded and visible regions into the same prototype, contaminating the body-axis sequence. SS2D then propagates this contamination. Observed: OCC-Duke ~58.2% vs PAM ~58.9%.

### 8. HFO-Lite (Prototype-based Soft Grouping)

Idea: GroupingUnit(cos_sigmoid) with K=3 part + K=2 fg/bg prototypes → soft assignment → part/fb features with ID+triplet+diverse loss.

Why failed: Prototype learning on 512-dim/128-token feature space is unstable. Grouping degenerates without HFO's original 2048-dim ResNet features and 4×Bottleneck1x1 post-processing. PAM+HFO did not exceed PAM baseline.

### 9. MGTP (Multi-Granularity Token Pooling)

Idea: Deterministic H-axis stripe pooling at 2/4/8 granularities, each with BNNeck+classifier.

Why failed: Same core issue as all stripe-based methods — deterministic geometric partitioning provides no learned adaptation to occlusion patterns. Auxiliary heads compete with global branch without adding complementary information.

### 10. TCS (Token Contribution Scoring)

Idea: Conv1x1(512→1)→Sigmoid to predict per-token importance, then weighted GeM pooling.

Status: Not yet implemented. Identified as potential next direction but superseded by literature review findings below.

---

## New Literature Review (2026-06-21)

Three 2024+ papers with open-source code directly relevant to token-level spatial attention for occluded ReID:

### DPEFormer (2024, Knowledge-Based Systems) ⭐ Top Pick

**Code**: `github.com/zhangxin06/DPEFormer`

Core module DPSM (Dynamic Patch Token Selection Module):
- Learnable proxy token → cosine similarity with each patch token → sigmoid score
- First-order derivative process for hard {0,1} token selection
- SAM-based occlusion augmentation + contrastive learning
- No external pose/parsing models needed

Key design insight: Uses a learnable proxy token as "what a good human token looks like" rather than learning per-token scores directly. More stable than direct Conv1x1 scoring.

### TSA (2024, Multimedia Tools and Applications)

Core module: Parameter-Free Token Spatial Attention
- Three weighting schemes: Mean (no occlusion), Variance (partial occlusion), Entropy (heavy occlusion)
- Variance weighting: occluded regions have low activation variance → naturally receive low weights
- Combined with Parallel Triplet Augmentation (essentially PAM)

Key design insight: Variance-based weighting is parameter-free, meaning it cannot degrade during training. Can serve as initialization or complementary signal to a learned scorer.

### DTST (2024, arXiv:2412.00433)

Core module VTS (Visual Token Selector):
- token_i → softmax(t_i^T W_q W_k^T t_i / √d) → importance score
- Gumbel-Softmax relaxation → differentiable Top-K selection
- Only K most informative tokens participate in downstream computation

Key design insight: Gumbel-Softmax enables end-to-end trainable hard selection, avoiding the "soft weighting averages out" problem.

---

## Deprecated Direction: DPEFormer-Inspired Token Scoring

Status update on 2026-06-26: this direction was effectively tested as R-PVTR and failed. Keep the notes below only as historical context. Do not prioritize the X1-X4 plan unless the token scorer is redesigned around explicit OSBBM mask/visible-token supervision.

Combining insights from DPEFormer, TSA, and the failed stripe/prototype routes:

### Core Design: Dual-Path Token Scoring for Weighted Pooling

```
feat_s4 (B, 512, 16, 8) = 128 tokens
    │
    ├── Path 1 (Learnable): Proxy Token Scoring
    │    proxy ∈ R^512 (learnable)
    │    score_proxy_i = sigmoid(cosine_sim(token_i, proxy) / τ)
    │    → importance_map (B, 1, 16, 8)
    │    ※ "Is this token semantically human-like?"
    │
    ├── Path 2 (Parameter-Free): Variance Gating
    │    var_i = variance of token_i across 512 channels
    │    score_var_i = sigmoid((var_i - mean(var)) / std(var))
    │    → variance_map (B, 1, 16, 8)
    │    ※ "Is this token informative or flat background?"
    │
    ├── score = score_proxy × score_var  (element-wise)
    │    Both must agree for high weight
    │
    └── feature × score → GeM Pooling → 512-dim
         → BNNeck → classifier (unchanged PAM heads)
```

### Why This Is Different From All Failed Attempts

| Property | Failed Methods | This Design |
|----------|---------------|-------------|
| Adds auxiliary branches/heads | Yes (stripe, HFO, MGTP) | **No** — only modifies pooling weights |
| Requires grouping tokens into parts | Yes (BPBreID, HFO, VBSM) | **No** — per-token scalar scores |
| Inference feature dimension changes | Yes (stripe concat, HFO concat) | **No** — always 512-dim |
| Can it degrade below baseline? | Yes (prototype collapse, gradient conflict) | **No** — worst case all scores≈1 → equals GeM |
| External supervision | PifPaf (BPBreID) | **None** — ID loss only |
| Parameter count | 0.5M-3M | **513** (1 proxy + 1 τ) |

### Design Rationale

1. **Single proxy over K prototypes**: HFO-lite's K=3+2 learned prototypes were unstable. One proxy token asking "is this human-like?" is a simpler, more stable question.

2. **Variance gating as safety net**: If proxy learning is unstable in early epochs, variance gating still provides meaningful weights (occluded regions → flat activations → low variance → low weight). This prevents early-training randomness from permanently biasing the proxy.

3. **Soft weighting over hard selection**: DPEFormer's hard {0,1} selection requires gradient tricks (first-order derivative). Soft weighting is naturally differentiable and allows partial contribution from uncertain tokens.

4. **No auxiliary loss needed**: The scoring is trained end-to-end through the existing ID+triplet loss. PAM's BA/CA/EA branches all use the same scorer (applied to their respective feature maps).

### Implementation Complexity

- New code: ~40 lines (1 class + proxy init + variance function)
- Integration: Insert between feat_s4 and GeM pooling in `_forward_pam`
- Config: 2 new keys (temperature τ, whether to use variance gate)
- No changes to loss, optimizer, or dataloader

### Historical Experiment Plan

| # | Config | Parameters | Validates |
|---|--------|-----------|-----------|
| X1 | PAM + variance-only scoring | 0 | TSA-style parameter-free weighting |
| X2 | PAM + proxy-only scoring | 513 | Learned proxy effectiveness |
| X3 | PAM + proxy × variance (full) | 513 | Complementary design |
| X4 | X3 + token diversity loss | 513 | Whether diversity helps scoring |

### Baselines To Compare Against

```text
Baseline 1: PAM(w=0.5), no OSBBM           → mAP 59.4, R1 68.3
Baseline 2: PAM(w=0.5) + OSBBM(21-120)     → mAP 60.0, R1 69.3
```

Success criterion: X3 > Baseline 2 by ≥0.3 mAP with stable epoch 160 performance.
---

## Historical MDB Direction: NIReID-Style Dual Branch Descriptor

Status update on 2026-06-26: MDB code/configs are not present in the current tracked code after rollbacks. Treat this section as historical context only.

After PLRM/PALC underperformed, the next implemented route is MDB
(Mamba Dual Branch Descriptor), inspired by NIReID dual branch and PADE-DES
local concat.

Core idea:

```text
MambaVision shared patch/stage1/stage2/stage3/main_proj
    -> branch_1: original stage4 -> BA global descriptor
    -> branch_2: copied stage4   -> second global + local descriptors
```

Inference descriptor:

```text
concat(
  branch1_gem,
  scale_g2  * branch2_gem,
  scale_max * branch1_max,
  scale_l   * branch2_local_parts
)
```

Implementation files:

- `model/make_model.py`
  - `MambaDualBranchDescriptor`
  - `MODEL.MDB` wiring
  - BA-only MDB descriptors during PAM training
- `loss/make_loss.py`
  - MDB auxiliary ID/triplet loss after BA/CA/EA PAM loss
- `solver/make_optimizer.py`
  - `mdb_branch2` uses `STAGE4_LR_FACTOR`
  - MDB heads use 2x LR
- `configs/OCC_Duke/mambavision_tiny_transreid_pam_padeaug_osbbm_mdb_b64k4.yml`
  - default recommended MDB setting
- `scripts/run_mdb_10groups.sh`
  - 10-group ablation script

10 groups:

| Group | Purpose |
|---|---|
| g00_baseline | Current PAM(w=0.5)+OSBBM baseline, MDB off |
| g01_b2g_only | Dual branch global descriptor only |
| g02_b2g_b1max | Add branch1 max descriptor |
| g03_l2_noloss | Add upper/lower local concat without local loss |
| g04_l2_ce005 | Add weak MDB CE loss, weight 0.05 |
| g05_l2_ce010 | Add MDB CE loss, weight 0.10 |
| g06_l2_ce_tri010 | Main candidate: CE+triplet on non-local MDB descriptors, weight 0.10 |
| g07_l2_ce_tri010_localtri | Also apply triplet to local parts |
| g08_l2_ce_tri020 | Stronger MDB loss, weight 0.20 |
| g09_l4_ce_tri010 | Four local parts, local scale 0.25 |

Run:

```bash
bash scripts/run_mdb_10groups.sh 0 1
```

Use multiple GPUs/jobs:

```bash
bash scripts/run_mdb_10groups.sh 0,1 2
```
