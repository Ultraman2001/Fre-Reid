# Fre-Reid / MambaVision + PAM Experiment Handoff

Date: 2026-06-10

This note summarizes the current project state, key code checkpoints, experiment results, and conclusions so a new conversation can continue quickly.

## Current Code State

The working code has been reverted to the **Base + PAM** line.

Core files restored to the Base+PAM commit:

- `config/defaults.py`
- `datasets/make_dataloader.py`
- `model/make_model.py`
- `loss/make_loss.py`
- `processor/processor.py`

The current active Base+PAM config is:

- `configs/OCC_Duke/mambavision_tiny_transreid_pam_b64k4.yml`

The CICO / EA_CONSIST experimental files were removed from the current working code:

- `utils/cico_pam.py`
- `configs/OCC_Duke/mambavision_tiny_transreid_pam_cico_b64k4.yml`
- `configs/OCC_Duke/mambavision_tiny_transreid_pam_eacons_b64k4.yml`

Important: the successful CICO append version is still recoverable from git.

## Key Git Checkpoints

Base+PAM checkpoint:

```text
1663ab4 feat: add PAM training for MambaVision ReID
```

Best CICO append checkpoint:

```text
c66de74 Add weak CICO regularizer for PAM
```

The CICO append commit corresponds to the best observed result:

```yaml
INPUT.PAM.CICO.ENABLED: True
INPUT.PAM.CICO.BRANCH_MODE: append
SOLVER.PAM_CICO_REID_LOSS_WEIGHT: 0.0
SOLVER.CICO_OCC_LOSS_WEIGHT: 0.1
SOLVER.CICO_OCC_LOSS_TYPE: cosine
```

## Dataset / Setting

Main dataset:

```text
Occluded-DukeMTMC
```

Main backbone:

```text
MambaVision-Tiny for TransReID / ReID
```

Main baseline setting:

```yaml
MODEL:
  TRANSFORMER_TYPE: mambavision_tiny_TransReID
  DROP_PATH: 0.5
  POOLING_TYPE: gem
  MAMBAVISION:
    GLOBAL_STAGES: [2, 3]

INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PAM:
    ENABLED: True
    CROP_PADDING: 30
    CROP_SCALE: [0.08, 1.0]
    CROP_RATIO: [0.75, 1.3333]

DATALOADER:
  NUM_INSTANCE: 4

SOLVER:
  MAX_EPOCHS: 160
  BASE_LR: 0.00035
  IMS_PER_BATCH: 64
  WARMUP_EPOCHS: 20
  PAM_AUGMENTED_LOSS_WEIGHT: 1.0
```

## PAM Implementation Details

Our current Base+PAM branch setup differs from PADE-main in augmentation details.

PADE-main actual transforms:

```text
BA = Resize + ToTensor + Normalize
CA = Resize + Pad(30) + ToTensor + Normalize + RandomResizedCrop(256,128)
EA = Resize + ToTensor + Normalize + RandomErasing(prob=1, mode=pixel)
```

In PADE-main, horizontal flip, pad+random crop for BA, and random erasing for BA are commented out.

Fre-Reid current Base+PAM transforms:

```text
BA = Resize + RandomHorizontalFlip(0.5) + ToTensor + Normalize
CA = Resize + RandomHorizontalFlip(0.5) + Pad(30)
     + RandomResizedCrop(scale=[0.08,1.0], ratio=[0.75,1.3333])
     + ToTensor + Normalize
EA = Resize + ToTensor + Normalize
     + RandomErasing(prob=1.0, mode=pixel, max_count=1)
```

So in our current PAM:

- BA has flip.
- CA has flip and irregular crop.
- EA has erase but no independent flip.

## Main Results

### Base+PAM

Result:

```text
Regular @160:
mAP 58.9
Rank-1 66.7
Rank-5 80.7
Rank-10 84.3

EMA @160:
mAP 59.3
Rank-1 67.2
Rank-5 81.0
Rank-10 84.8
```

Conclusion:

```text
Strong and stable baseline, but by itself not enough as a paper contribution.
```

### PAM loss weight 0.7

Setting:

```yaml
PAM_AUGMENTED_LOSS_WEIGHT: 0.7
```

Result:

```text
Regular @160:
mAP 58.9
Rank-1 66.8
Rank-5 80.0
Rank-10 84.5

EMA @160:
mAP 59.2
Rank-1 67.2
Rank-5 79.8
Rank-10 84.9
```

Conclusion:

```text
No meaningful improvement over Base+PAM.
```

### HFO-lite / HDFG Attempts

Several HFO-lite and HDFG-inspired variants were tested.

Representative result:

```text
HDFG with weights:
HDFG_OBJ_BA_LOSS_WEIGHT: 0.1
HDFG_PART_BA_LOSS_WEIGHT: 0.1
HDFG_OBJ_CA_LOSS_WEIGHT: 0.0

@160:
mAP 58.1
Rank-1 66.6
Rank-5 79.3
Rank-10 84.2
```

Earlier stronger HDFG setting:

```text
@160:
mAP 56.9
Rank-1 66.1
Rank-5 79.5
Rank-10 82.8
```

Conclusion:

```text
HFO-lite/HDFG did not outperform Base+PAM.
Likely issue: local/prototype constraints conflict with current MambaVision+PAM training.
Stopped.
```

### LRCA

Result:

```text
Base+PAM+LRCA mAP around 58.0
```

Conclusion:

```text
Below Base+PAM 58.9. Stopped.
```

### COPE/CICO Append - Best Variant

Implemented based on `COPE-main` CICO inspiration.

Important distinction:

```text
COPE full system = CICO + PBF + segmentation/mask + memory + multiple alignments.
Our adapted CICO append = weak auxiliary CICO branch with occlusion consistency only.
```

Best setting:

```yaml
BRANCH_MODE: append
PAM_CICO_REID_LOSS_WEIGHT: 0.0
CICO_OCC_LOSS_WEIGHT: 0.1
CICO_OCC_LOSS_TYPE: cosine
```

Result:

```text
Regular @160:
mAP 59.6
Rank-1 67.3
Rank-5 81.1
Rank-10 85.2
```

Conclusion:

```text
Best observed result.
CICO helps as a weak auxiliary occlusion-consistency regularizer.
CICO ReID loss should be disabled.
```

Recover from:

```text
git checkout c66de74
```

or selectively restore the files from that commit.

### CICO Append - Strong Loss Failed

Setting:

```yaml
PAM_CICO_REID_LOSS_WEIGHT: 1.0
CICO_OCC_LOSS_WEIGHT: 1.0
```

Result:

```text
@160:
mAP 56.5
Rank-1 64.7
Rank-5 77.5
Rank-10 82.3
```

Conclusion:

```text
Too strong. Four-branch CICO with ReID loss hurts identity learning.
```

### CICO Replace-EA

Settings and results:

```text
replace_ea, CICO_REID=0.0, OCC=0.1:
mAP 56.8, Rank-1 66.2

replace_ea, CICO_REID=0.3, OCC=0.0:
mAP 57.9, Rank-1 65.6

replace_ea, CICO_REID=0.3, OCC=0.1:
mAP 58.2, Rank-1 66.6
```

Conclusion:

```text
CICO cannot replace EA. Original random erasing is important.
```

### EA-CICO

EA-CICO idea:

```text
BA + CA + EA, but EA uses group-shared CICO occlusion.
```

Result:

```text
@120:
mAP only around 55.x
```

Conclusion:

```text
Failed. It effectively replaced random EA with CICO patch and destroyed EA diversity.
Stopped.
```

### EA-Consistency

EA-consistency idea:

```text
Keep BA + CA + EA three branches.
EA keeps random erasing, but group-shared erase positions are used.
Apply consistency loss on erased-region features.
```

Settings tested:

```yaml
EA_CONSIST_LOSS_WEIGHT: 0.05
```

Result:

```text
@140:
mAP 58.7
Rank-1 67.8

@160:
mAP 58.3
Rank-1 66.4
```

Conclusion:

```text
Too weak / no final gain.
```

Setting:

```yaml
EA_CONSIST_LOSS_WEIGHT: 0.5
```

Result:

```text
@140:
mAP 59.6
Rank-1 68.0
Rank-5 81.2
Rank-10 85.7

@160:
mAP 59.1
Rank-1 66.7
Rank-5 81.0
Rank-10 85.4
```

Setting:

```yaml
EA_CONSIST_LOSS_WEIGHT: 1.0
```

Result:

```text
@140:
mAP 59.1
Rank-1 67.2
Rank-5 81.4
Rank-10 85.9

@160:
mAP 58.7
Rank-1 66.7
Rank-5 80.3
Rank-10 85.2
```

Later attempted stop control:

```yaml
EA_CONSIST_LOSS_WEIGHT: 0.5
EA_CONSIST_STOP_EPOCH: 140
```

Result:

```text
@140:
mAP 58.8

@160:
mAP 58.6
```

Conclusion:

```text
EA-consistency is unstable. It can create mid-training peaks but does not reproduce reliably.
Stopped and reverted.
```

## Important Implementation Lessons

1. CICO as an auxiliary append branch worked only when ReID loss on CICO was disabled.

2. CICO replacing EA failed. EA random diversity is necessary.

3. EA-consistency can create a mid-training boost but is unstable and not reliable enough.

4. Strong local/prototype losses repeatedly underperformed Base+PAM.

5. Current strongest reliable code line is Base+PAM.

6. Current strongest experimental result is Base+PAM+CICO append from commit `c66de74`.

## Current Recommendation

Short-term:

```text
Use Base+PAM as clean stable code.
Keep c66de74 as the best experimental branch for CICO append.
Do not continue EA-CICO / EA-consistency unless a new theoretical reason appears.
```

For future paper contribution:

```text
Base+PAM alone is not enough.
The most promising direction is not more augmentation branches,
but a lightweight module that uses local/occlusion cues without hurting global ranking.
```

Possible next directions:

1. Return to `c66de74` and analyze why weak CICO append works.
2. Add a control experiment:

```yaml
BRANCH_MODE: append
PAM_CICO_REID_LOSS_WEIGHT: 0.0
CICO_OCC_LOSS_WEIGHT: 0.0
```

This checks whether the 59.6 improvement is from the consistency loss or simply from extra branch computation.

3. Consider a DES-like local feature module inspired by PADE, but avoid heavy local supervision that previously hurt HFO/HDFG.

4. If adding local modules, keep inference single/global feature unless the module clearly improves mAP.

## Notes for New Conversation

If continuing from Base+PAM:

```text
Use current working tree.
Run configs/OCC_Duke/mambavision_tiny_transreid_pam_b64k4.yml.
```

If continuing from best CICO append:

```text
Use commit c66de74.
```

If comparing with PADE-main:

```text
PADE-main BA/CA/EA transforms are simpler than ours.
PADE-main has horizontal flip commented out in all three branches.
Our Fre-Reid PAM uses flip in BA and CA; EA has no independent flip unless derived from BA in the discarded EA_CONSIST variant.
```

Do not treat FECR as PADE. They are different projects.
