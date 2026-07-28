# Fre-Reid Project Handoff

Created: 2026-07-15
Updated: 2026-07-27 after ODSMF20 analysis and pure-state follow-up
Scope: MambaVision + OSNet + FCU + FDMF + semantic-detail local branch

This document is the current entry point for the project. It supersedes
`EXPERIMENT_HANDOFF_2026-06-10.md`, which records the older Base+PAM phase.

## 1. One-Minute Status

The current research line is a heterogeneous dual-branch ReID network:

```text
MambaVision branch (global context)
        <-> stage-aware FCU interaction
OSNet branch (omni-scale convolutional detail)
        -> FDMF heterogeneous feature fusion
        -> main retrieval descriptor
Mamba Stage3 + OSNet conv4 -> semantic-detail auxiliary local supervision
```

The architecture is already large enough to support a paper. Do not add more
unrelated modules. The latest evidence has moved the project from broad local
and augmentation searches to a clean component audit of FDMF.

Current decisions:

- Keep the original bidirectional raster scan as the default FDMF scan.
- Keep `axial4` and `snake4` as optional ablations only.
- Keep OSBBM disabled in the formal baseline.
- Keep RATR disabled; it belongs to the first paper and did not help here.
- Do not force branches to be orthogonal or completely non-redundant.
- Do not use heterogeneous input views as the formal method. Shared aligned
  random erasing remains the reliable training convention.
- Treat the semantic-detail local branch primarily as a training-time auxiliary
  branch. Its direct descriptor contribution is consistently only `0-0.1 mAP`.
- Stage3 FCU is not a cross-dataset winner. Duke currently favors Stage1+2;
  Stage2-only is the conservative cross-dataset structure until Stage1 is
  validated outside Duke.
- The abandoned FDMF `role`/`FCC` variants have been removed from the code.
- Evaluate new searches at epoch 160 only unless an instability diagnosis is needed.

Current Duke structure candidate:

```text
Stage1+2 FCU + semantic-detail local supervision + no local descriptor
seed42:   84.9 mAP / 91.7 R1 / 96.6 R5 / 97.6 R10
seed3407: 84.8 mAP / 92.1 R1 / 96.3 R5 / 97.3 R10
mean:     84.85 mAP / 91.90 R1
```

This is the strongest Duke hierarchy result, not yet a cross-dataset final
architecture. The older Stage2+3 semantic-detail descriptor result remains a
valid historical checkpoint result but is no longer the preferred structural
interpretation.

## 2. Working Tree Warning

The worktree is intentionally dirty. Many core files and scripts are modified
or untracked. Do not reset, checkout, or remove changes without inspecting them.

Important modified files include:

- `config/defaults.py`
- `model/make_model.py`
- `loss/make_loss.py`
- `loss/triplet_loss.py`
- `processor/processor.py`
- `solver/make_optimizer.py`
- `utils/osbbm.py`
- Duke, Market, MSMT17, and OCC-Duke FDMF configs

Important new scripts include:

- `scripts/run_duke_fdmf_complementarity_train24.sh`
- `scripts/run_duke_fdmf_hypothesis20.sh`
- `scripts/run_duke_fdmf_stage3_local24.sh`
- `scripts/run_duke_fdmf_stage3_local_shortlist.sh`
- `scripts/run_duke_fdmf_semantic_detail4.sh`
- `scripts/run_duke_fdmf_semantic_detail20.sh`
- `scripts/run_duke_fdmf_semantic_detail20_seed3407.sh`
- `scripts/test_duke_fdmf_semantic_detail_best_infer_sweep.sh`
- `scripts/run_duke_fdmf_semantic_detail_osbbm16.sh`
- `scripts/run_duke_fdmf_scan4.sh`
- `scripts/run_fdmf_scan8_duke_market.sh`
- `scripts/run_duke_fdmf_local24.sh`
- `scripts/test_duke_fdmf_local24_infer_decomposition.sh`
- `scripts/run_duke_fdmf_local_guided6.sh`
- `scripts/run_duke_fdmf_local_causal8.sh`
- `scripts/run_duke_fdmf_s3fcu_local_factorial4.sh`
- `scripts/run_duke_fdmf_stage1_fcu_audit8.sh`
- `scripts/run_duke_fdmf_dual_view28.sh`
- `scripts/run_duke_fdmf_role_specialization32.sh`
- `scripts/run_duke_fdmf_component6.sh`

## 3. Current Network

### 3.1 Inputs and feature maps

- Input resolution: `256 x 128`.
- The final FDMF spatial feature map is expected to be `16 x 8`.
- The design operates on feature maps. Do not propose token-only multi-layer
  Transformer interaction that discards this structure.

### 3.2 Main branches

MambaVision-Tiny is the global/context branch. OSNet-x1.0 is the convolutional
detail branch. Both are ImageNet pretrained and trained jointly.

Current branch-loss settings in the formal YAML files:

```yaml
Mamba loss weight: 1.0 (implicit in loss code)
OSNET_LOSS_WEIGHT: 0.5
FUSED_LOSS_WEIGHT: 1.0
STAGE3_STRIPE_LOCAL_LOSS_WEIGHT: 0.1
```

The branch losses are normalized by their sum. Therefore the local branch has
only `0.1 / 2.6 = 3.85%` of the aggregate ID and triplet supervision.

### 3.3 Stage-aware FCU

Previously formalized interaction:

```yaml
FCU_STAGES: [2, 3]
FCU_DIRECTION: bidirectional
FCU_STAGE2_DIRECTION: osnet_to_mamba
FCU_STAGE3_DIRECTION: mamba_to_osnet
FCU_INIT_SCALE: 0.1
```

Historical interpretation:

- Stage2 injects convolutional detail into MambaVision.
- Stage3 injects stronger global semantics back into OSNet.
- This asymmetric stage direction was a central candidate contribution.

Latest hierarchy evidence supersedes the claim that `[2,3]` is the best FCU
layout. Across two seeds, Stage3 gives only `+0.15 mAP/0.00 R1` on Duke and
`-0.25 mAP/-0.30 R1` on Market. Duke Stage1+2 gives `84.9/91.7` and
`84.8/92.1` across seeds and is the current Duke candidate. Stage1 has not yet
been validated on Market/MSMT17, so the conservative cross-dataset fallback is
Stage2-only.

### 3.4 FDMF

Current FDMF settings:

```yaml
FUSION_TYPE: fdmf
FUSION_NORM: none
FDMF_FUSED_FORM: mamba_fdmf
FDMF_MAMBA_DEPTH: 1
FDMF_MAMBA_D_STATE: 8
FDMF_MAMBA_D_CONV: 3
FDMF_MAMBA_BIDIRECTIONAL: true
FDMF_MAMBA_SCAN_MODE: raster (default from config/defaults.py)
FDMF_MSEF_ENABLED: true
FDMF_MSEF_REDUCTION_RATIO: 16
```

The raster mode flattens the `16 x 8` map row by row and scans the resulting
128-position sequence in forward and reverse order. The two restored outputs
are averaged. The optional `axial4` and `snake4` modes use four paths.

### 3.5 Semantic-detail local branch

Current best structure:

```yaml
STAGE3_STRIPE_LOCAL_ENABLED: true
STAGE3_LOCAL_TYPE: semantic_detail
STAGE3_STRIPE_LOCAL_NUM_STRIPES: 2
STAGE3_STRIPE_LOCAL_MAMBA_DEPTH: 0
STAGE3_STRIPE_LOCAL_LOSS_WEIGHT: 0.1
STAGE3_STRIPE_LOCAL_INFER_WEIGHT: 0.0 for the latest mechanism/FCU/component audits
STAGE3_SOFT_TEMPERATURE: 1.5
STAGE3_SOFT_PRIOR_SCALE: 1.0
STAGE3_DETAIL_MASK_STAGE: stage3
STAGE3_DETAIL_FOREGROUND_STAGE: stage3
STAGE3_DETAIL_SOURCE: conv4
STAGE3_DETAIL_FOREGROUND_GATE: true
STAGE3_DETAIL_RESIDUAL_INJECTION: false
STAGE3_LOCAL_GUIDED_TRIPLET_MIX: 0.75 in latest hierarchy/component controls
STAGE3_LOCAL_DETACH_PROMPT: true in latest hierarchy/component controls
STAGE3_LOCAL_DETACH_DETAIL: false
```

Logic:

1. Mamba Stage3 generates two semantic part masks.
2. Mamba Stage3 also generates foreground guidance.
3. OSNet conv4 provides the detail feature map.
4. The masks pool the gated detail map into two local part descriptors.
5. The two parts are flattened into one local descriptor.

Current implementation status:

- Separate local ID and triplet weights are implemented and were tested.
- Optional part-specific ID heads, part-average triplet and confidence modes are
  implemented, but none beat the simple flattened local supervision.
- The retained foreground gate is `detail * (0.5 + sigmoid)`; suppressive and
  `sigmoid2` alternatives did not provide a stable mAP advantage.
- Balance/order regularizers control mass and vertical order. Stronger
  regularization won one seed by only `0.1 mAP` and is not a central claim.

Current interpretation:

- Local supervision improves the separately trained checkpoint by roughly
  `+0.2 to +0.5 mAP` in the recent controlled searches.
- Adding the local descriptor back at inference contributes only `0 to +0.1
  mAP` and sometimes reduces R1.
- Therefore the latest main descriptor excludes the local descriptor. The
  branch is retained as semantic-guided auxiliary supervision.

Configuration warning:

```text
The formal YAML still contains historical `STAGE3_STRIPE_LOCAL_INFER_WEIGHT`
values, while the latest scripts explicitly override the value to `0.0`.
Always check the merged command-line config. In particular, the `0.3` used in
local24/guided searches was a controlled diagnostic value, not evidence that
`0.3` should be used in the final descriptor.
```

### 3.6 Optimizer and inference

Current optimizer choices:

```yaml
OPTIMIZER_NAME: AdamW
BASE_LR: 0.00035
OSNET_LR_FACTOR: 2.0
OSNET_WEIGHT_DECAY: 0.0005
OSNET_WEIGHT_DECAY_BIAS: 0.0005
OSNET_FUSION_LR_FACTOR: 3.0
CLIP_GRAD_NORM: 10.0
```

Historical semantic-detail inference descriptor:

```yaml
TEST.FEAT_MODE: weighted_mamba_fdmf_osnet_stage3local
STAGE3_STRIPE_LOCAL_INFER_WEIGHT: 0.1
TEST.FEAT_NORM: yes
```

Latest mechanism/FCU/FDMF component experiments use:

```yaml
TEST.FEAT_MODE: weighted_mamba_fdmf_osnet
STAGE3_STRIPE_LOCAL_INFER_WEIGHT: 0.0
```

## 4. Baselines: Avoid This Naming Trap

There are several baselines from different experiment phases.

### 4.1 Complementarity-24 Stage A winner

Within that specific search space, A01 won:

```text
OSNet loss weight = 0.25
FDMF loss weight = 1.0
no joint loss, no Branch Drop
83.95 +- 0.25 mAP / 91.50 +- 0.10 R1
```

This is only the winner of the Complementarity-24 Stage A search. It is not the
current overall project baseline.

### 4.2 Historical Stage2+3 semantic-detail checkpoint baseline

The dataset YAML baseline uses:

```text
Mamba loss = 1.0
OSNet loss = 0.5
FDMF loss = 1.0
OSNet LR factor = 2.0
OSNet WD = 0.0005
semantic-detail local enabled
OSBBM disabled
```

Canonical dual-seed results from the earlier inference sweep:

| Seed | Local infer weight | mAP | R1 | R5 | R10 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.1 | 84.7 | 92.3 | 96.1 | 97.3 |
| 3407 | 0.1 | 84.8 | 92.5 | 96.5 | 97.2 |
| Mean | 0.1 | 84.75 | 92.40 | 96.30 | 97.25 |

The recorded No-local seed3407 result was:

```text
83.8 mAP / 91.7 R1 / 95.7 R5 / 97.2 R10
```

One later Duke YAML training run reported validation `84.4/91.7`, while a
separate test invocation reported `84.7/92.3`. Before a paper table is frozen,
audit checkpoint paths and exact inference options so results are provenance-clean.

### 4.3 Latest controlled baseline

The latest controlled experiments freeze:

```text
shared aligned random erasing p=0.75
Stage1+2 FCU, OSNet -> Mamba at both stages (Duke candidate)
semantic-detail local supervision enabled
guided triplet mix=0.75 and detach prompt=true in hierarchy/component audits
local descriptor excluded at inference
OSBBM/PAM/role-specialization disabled
```

The stable result to quote for this Duke structure is the two-seed Stage1+2
mean `84.85 mAP / 91.90 R1`, not the anomalous single A00 value
`84.9/92.3`. The exact A00 Stage2+3 replicas were `84.7/92.4` and
`84.2/91.7`, giving `84.45/92.05` with a large `0.50/0.70` spread.

## 5. Experiment Record

### 5.1 Complementarity-24

Files:

- `scripts/run_duke_fdmf_complementarity_train24.sh`
- `logs/summary/summary_a.txt`
- `logs/summary/summary_b.txt`
- `logs/summary/summary_c.txt`
- `logs/summary/summary_all.txt`

Stage A tested branch loss allocation and selected A01 at `83.95/91.50` mean.

Stages B and C failed badly:

- Joint descriptor supervision caused late-training collapse.
- Branch Drop did not rescue it.
- Raw covariance, `shared_m`, and `shared_mo` complementarity objectives also
  degraded severely.
- Example: Stage B best by its internal ranking was only `68.15/84.15` at epoch
  160; Stage C best was only `77.00/88.45`.

Conclusion:

```text
Do not force branch features to become fully different.
Mamba and FDMF legitimately share identity information.
Hard redundancy removal destroys useful shared representation.
```

### 5.2 Peer complementarity and hypothesis-20

Files:

- `logs/summary.txt`
- `logs/summary20.txt`
- `scripts/run_duke_fdmf_peer_complement6.sh`
- `scripts/run_duke_fdmf_hypothesis20.sh`

Peer complementarity did not beat its two-seed base:

| Setting | Mean mAP | Mean R1 |
|---|---:|---:|
| Base | 84.00 | 91.45 |
| Fused metric | 83.95 | 91.25 |
| Peer complement | 83.90 | 91.35 |

Hypothesis-20 key result:

| Setting | Mean mAP | Mean R1 |
|---|---:|---:|
| Base | 83.90 | 91.30 |
| OSNet LR 2.0, WD 0.0005 | **84.10** | **91.80** |
| OSBBM 8/2 | 83.95 | 91.20 |
| RATR variants | <=83.90 | <=91.25 |

Conclusion:

- OSNet benefits from `2x` LR and small OSNet-specific weight decay.
- RATR is not required and should remain disabled.
- Peer losses did not provide reliable complementarity.

### 5.3 Local24 and two-seed shortlist

Files:

- `logs/summary_s42.txt`
- `logs/summary_shortlist_2seed.txt`
- `scripts/run_duke_fdmf_stage3_local24.sh`
- `scripts/run_duke_fdmf_stage3_local_shortlist.sh`

Seed42 no-local was `84.1/91.4`. The broad local search showed mostly small
changes. Representative final results:

| Variant | mAP | R1 | Delta vs no-local |
|---|---:|---:|---:|
| soft Mamba mask, k2 | 84.2 | 91.9 | +0.1 / +0.5 |
| soft OSNet, k2, depth2 | 84.2 | 91.7 | +0.1 / +0.3 |
| soft OSNet, T=1.5 | 84.1 | 92.1 | +0.0 / +0.7 |
| hard k2 depth1 | 84.1 | 91.9 | +0.0 / +0.5 |

Two-seed shortlist:

| Variant | Mean mAP | Mean R1 |
|---|---:|---:|
| No-local | 84.00 | 91.60 |
| Hard k2 depth1 | 84.20 | 91.55 |
| Soft Mamba k2 | 84.05 | 91.60 |
| Soft OSNet T=1.5 | 84.00 | 91.80 |

Two unstable runs were observed:

- `l24_hard_k2_d0_s42`
- `l24_soft_os_k2_d2_s3407`

The latter was successfully retrained after numerical stability fixes and
reached `84.1/91.7` at epoch 160.

### 5.4 Semantic-detail local search

Files:

- `logs/summarydetail4.txt`
- `logs/summary_sdetail20.txt`
- semantic-detail scripts listed above

The first four experiments established that a semantic mask over a detail
feature map was viable. The 20-way search then identified:

```text
mask = Mamba Stage3
foreground = Mamba Stage3
detail = OSNet conv4
parts = 2
temperature = 1.5
prior = 1.0
no residual injection
```

Seed42 result:

```text
84.7 mAP / 92.2 R1 / 96.2 R5 / 97.3 R10
```

Seed3407 reproduction:

```text
84.8 mAP / 92.4 R1 / 96.5 R5 / 97.1 R10
```

Important observations:

- OSNet conv4 detail consistently beat raw conv2 detail. Semantic quality was
  more important than resolution alone.
- Mamba Stage3 was the best mask and foreground source.
- Two parts were sufficient; more parts were not consistently better.
- Strong prior scale `2.0` failed badly (`78.8/88.3`).
- Residual detail injection was not helpful.

### 5.5 Local inference-weight sweep

File: `scripts/test_duke_fdmf_semantic_detail_best_infer_sweep.sh`

Two-seed means:

| Local infer weight | Mean mAP | Mean R1 |
|---:|---:|---:|
| 0.0 | 84.65 | 92.40 |
| 0.1 | **84.75** | **92.40** |
| 0.2 | 84.75 | 92.30 |
| 0.3 | 84.75 | 92.30 |
| 0.5 | 84.75 | 92.20 |
| 0.7 | 84.70 | 92.05 |

Conclusion:

- Keep inference weight at `0.1`.
- The local descriptor itself contributes only about `+0.1 mAP` on these
  checkpoints.
- Much of the gain over a separately trained no-local model may be auxiliary
  training regularization rather than direct descriptor concatenation.

### 5.6 Cross-dataset local validation

Market-1501:

| Setting | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| Semantic-detail full descriptor | 90.1 | 95.5 | 98.4 | 99.2 |
| Same checkpoint without local descriptor | 90.1 | 95.6 | 98.4 | 99.2 |
| Separately trained no-local model | 89.7 | 95.2 | 98.2 | 98.9 |

MSMT17:

| Setting | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| Semantic-detail full descriptor | 68.5 | 85.6 | 92.1 | 93.9 |
| Same checkpoint without local descriptor | 68.4 | 85.6 | 92.1 | 93.9 |
| Historical separately trained no-local | 68.1 | 85.5 | 92.1 | 94.0 |

Conclusion:

- The local branch is clearly useful on Duke.
- Its direct descriptor contribution is negligible on Market and MSMT17.
- It may still improve shared feature training slightly.
- Do not claim universally strong local-descriptor gains yet.

### 5.7 OSBBM-16

File: `scripts/run_duke_fdmf_semantic_detail_osbbm16.sh`

External No-OSBBM baseline for that experiment was `84.7/92.3`.

Completed representative results:

| Variant | mAP | R1 |
|---|---:|---:|
| Legacy always, gray 0.5 | 84.1 | 91.8 |
| Legacy always, gray 0.0 | 84.5 | 91.2 |
| Range epochs 21-120 | 84.6 | 91.8 |
| Cycle epochs 21-120 | 84.4 | 91.7 |
| Cycle, mixed gray scope | **84.7** | **92.0** |
| Cycle, mix 3 blocks | 84.3 | 91.7 |
| PK-half sampling | 84.5 | 91.7 |
| Derangement donor | 84.6 | 91.5 |
| Part-balanced blocks | 84.4 | 91.7 |
| Structured combination | 84.4 | 92.0 |

Several variants crashed or did not produce final results, including
`os16_range41_120`. Non-finite loss was detected with OSBBM active in at least
one run, but a similar Market crash occurred with OSBBM disabled, so OSBBM is
not the sole numerical cause.

Conclusion:

- OSBBM still has research space, but no tested variant beats the current
  No-OSBBM baseline.
- Keep it out of the main architecture for now.
- New controls added during this phase include schedule, PK-half sampling,
  deranged donors, balanced body blocks, gray scope, and mixed-label handling.

### 5.8 FDMF scan search

Files:

- `scripts/run_duke_fdmf_scan4.sh`
- `scripts/run_fdmf_scan8_duke_market.sh`

Results:

| Dataset | Scan | Learnable weights | mAP | R1 | Delta mAP/R1 |
|---|---|---:|---:|---:|---:|
| Duke | raster | no | 84.4 | 91.4 | +0.0 / +0.0 |
| Duke | axial4 | no | 84.5 | 91.7 | +0.1 / +0.3 |
| Duke | snake4 | no | 84.5 | 92.0 | +0.1 / +0.6 |
| Duke | snake4 | yes | 84.5 | 91.9 | +0.1 / +0.5 |
| Market | raster | no | 90.1 | 95.5 | +0.0 / +0.0 |
| Market | axial4 | no | 90.0 | 95.5 | -0.1 / +0.0 |
| Market | snake4 | no | 90.1 | 95.5 | +0.0 / +0.0 |
| Market | snake4 | yes | 90.1 | 95.5 | +0.0 / +0.0 |

Cross-dataset mean delta for fixed snake4 was only `+0.05 mAP/+0.30 R1`.

Decision:

```text
Default: bidirectional raster
Optional ablation: axial4, snake4
Do not use learnable direction weights by default
```

Four paths approximately double the Mamba mixer work relative to the two-path
raster mode, without a stable Market gain.

### 5.9 Local supervision 24 and inference decomposition

Files:

- `scripts/run_duke_fdmf_local24.sh`
- `scripts/test_duke_fdmf_local24_infer_decomposition.sh`

Provenance warning: these results were returned from the server in this project
thread. The server summary was not copied into this worktree. Local
`logs/summary_s42.txt` is the older broad stripe-local search in Section 5.3,
not this `l24_e00-e23` experiment.

All 24 Duke seed42 results (`iw=0.3` during the original summary):

| Experiment | Family | ID | Tri | Part ID | Tri mode | Confidence | Gate | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|
| e00 nolocal std | control | 0 | 0 | none | flat | none | emphasis | 84.3 | 91.7 | 96.0 | 97.1 |
| e01 nolocal d2.6 | control | 0 | 0 | none | flat | none | emphasis | 84.1 | 90.9 | 95.9 | 97.3 |
| e02 reg-only | control | 0 | 0 | none | flat | none | emphasis | 83.4 | 91.2 | 96.1 | 97.4 |
| e03 id10 tri10 | loss | .1 | .1 | none | flat | none | emphasis | 84.6 | 91.8 | 96.2 | 97.3 |
| e04 id10 tri00 | loss | .1 | 0 | none | flat | none | emphasis | 84.3 | 92.0 | 95.9 | 97.3 |
| e05 id00 tri10 | loss | 0 | .1 | none | flat | none | emphasis | 84.6 | 92.1 | 96.0 | 97.2 |
| e06 id10 tri05 | loss | .1 | .05 | none | flat | none | emphasis | 84.6 | 92.0 | 96.3 | 97.3 |
| e07 id20 tri10 | loss | .2 | .1 | none | flat | none | emphasis | 84.6 | 92.0 | 96.0 | 97.2 |
| e08 id20 tri05 | loss | .2 | .05 | none | flat | none | emphasis | 84.6 | 92.0 | 96.1 | 97.2 |
| e09 id30 tri10 | loss | .3 | .1 | none | flat | none | emphasis | 84.6 | 92.2 | 96.3 | 97.4 |
| e10 id30 tri05 | loss | .3 | .05 | none | flat | none | emphasis | 84.4 | 92.0 | 96.1 | 97.1 |
| e11 id10 tri20 | loss | .1 | .2 | none | flat | none | emphasis | 84.4 | 91.9 | 96.2 | 97.2 |
| e12 no regularizer | regularizer | .1 | .1 | none | flat | none | emphasis | 84.5 | 92.1 | 96.1 | 97.2 |
| e13 balance only | regularizer | .1 | .1 | none | flat | none | emphasis | 84.6 | 91.9 | 96.3 | 97.1 |
| e14 order only | regularizer | .1 | .1 | none | flat | none | emphasis | 84.6 | 91.8 | 96.1 | 97.2 |
| e15 strong regularizer | regularizer | .1 | .1 | none | flat | none | emphasis | **84.7** | 92.2 | 96.1 | 97.0 |
| e16 part-ID flat-triplet | part | .1 | .1 | replace | flat | none | emphasis | 84.2 | 91.7 | 95.9 | 96.9 |
| e17 joint part-ID | part | .1 | .1 | joint | flat | none | emphasis | 84.3 | 91.2 | 95.8 | 96.9 |
| e18 flat-ID part-avg-triplet | part | .1 | .1 | none | part_avg | none | emphasis | 84.6 | 91.7 | 96.0 | 97.1 |
| e19 part-ID part-avg-triplet | part | .1 | .1 | replace | part_avg | none | emphasis | 84.3 | 91.7 | 96.2 | 97.0 |
| e20 confidence triplet | confidence | .1 | .1 | none | part_avg | triplet | emphasis | 84.5 | 91.8 | 96.0 | 97.2 |
| e21 confidence descriptor | confidence | .1 | .1 | none | part_avg | descriptor | emphasis | 84.5 | 92.0 | 95.9 | 97.3 |
| e22 gate sigmoid2 | gate | .1 | .1 | none | flat | none | sigmoid2 | 84.6 | **92.3** | 96.1 | 97.2 |
| e23 gate floor01 | gate | .1 | .1 | none | flat | none | floor01 | 84.5 | 91.7 | 96.1 | 97.2 |

Inference decomposition for selected checkpoints:

Each tuple is `mAP/R1/R5/R10`:

| Experiment | iw0 | iw.1 | iw.3 | Local-only |
|---|---|---|---|---|
| e02 reg-only | 84.2/91.6/96.3/97.4 | 84.2/91.7/96.2/97.4 | 83.4/91.2/96.1/97.4 | 22.8/40.6/54.8/61.1 |
| e03 id10 tri10 | 84.6/92.0/96.1/97.3 | 84.6/92.0/96.1/97.3 | 84.6/91.8/96.2/97.3 | 78.1/88.0/94.1/96.0 |
| e05 id00 tri10 | 84.5/92.1/96.0/97.2 | 84.5/92.1/95.9/97.2 | 84.6/92.1/96.0/97.2 | 77.4/88.6/94.0/96.0 |
| e15 strong regularizer | 84.6/92.1/96.2/97.0 | 84.6/92.1/96.2/97.0 | 84.7/92.2/96.1/97.0 | 78.3/88.9/94.5/96.0 |
| e22 sigmoid2 | 84.5/92.1/96.1/97.1 | 84.5/92.2/96.1/97.1 | 84.6/92.3/96.1/97.2 | 78.6/88.3/95.0/96.4 |

| Experiment | Train delta @iw0 vs e00 (mAP/R1) | Direct iw.3-iw0 (mAP/R1) | Local-only mAP/R1 |
|---|---:|---:|---:|
| e02 reg-only | -0.1 / -0.1 | -0.8 / -0.4 | 22.8 / 40.6 |
| e03 id10 tri10 | +0.3 / +0.3 | 0.0 / -0.2 | 78.1 / 88.0 |
| e05 id00 tri10 | +0.2 / +0.4 | +0.1 / 0.0 | 77.4 / 88.6 |
| e15 strong regularizer | +0.3 / +0.4 | +0.1 / +0.1 | 78.3 / 88.9 |
| e22 sigmoid2 | +0.2 / +0.4 | +0.1 / +0.2 | 78.6 / 88.3 |

Conclusion: local ID/triplet supervision changes shared representation and is
the source of most gain. Part-specific heads, part-average triplet and
confidence weighting did not help. A local branch with regularizers but no
identity/metric supervision failed to learn a useful descriptor.

### 5.10 Guided-local and causal follow-up

Guided6 tested the fraction of local triplet mining guided by the main
descriptor:

| Experiment | Guided mix | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| no-local | 0 | 84.2 | 91.5 | 96.0 | 97.2 |
| local self | 0 | 84.5 | 91.4 | 96.2 | 97.2 |
| guided .25 | .25 | 84.5 | 91.8 | 96.1 | 97.3 |
| guided .50 | .50 | 84.6 | 91.8 | 96.0 | 97.2 |
| guided .75 | .75 | **84.7** | **91.9** | 96.3 | 97.4 |
| guided 1.00 | 1.00 | 84.5 | 91.7 | 96.4 | 97.4 |

The seed3407 Causal8 did not reproduce a clear `.75` mAP advantage:

Each metric tuple below is `mAP/R1/R5/R10`:

| Experiment | iw0 | iw.1 | iw.3 | Local-only |
|---|---|---|---|---|
| e01 local self | 84.6/91.5/96.1/97.4 | 84.6/91.4/96.1/97.4 | 84.6/91.6/96.1/97.5 | 78.4/88.6/95.2/96.0 |
| e02 guided .75 full | 84.4/91.8/96.0/96.9 | 84.4/91.8/96.0/96.9 | 84.4/91.8/95.9/96.9 | 77.0/87.9/94.2/95.6 |
| e03 detach prompt | 84.5/92.2/96.2/97.4 | 84.5/92.2/96.2/97.4 | 84.5/92.2/96.3/97.5 | 77.1/87.9/94.2/95.7 |
| e04 detach detail | 84.1/91.0/96.2/97.0 | 84.1/91.0/96.1/97.0 | 84.2/91.1/96.1/97.1 | 71.8/85.1/92.8/94.7 |
| e05 detach both | 84.5/91.7/96.3/97.5 | 84.5/91.6/96.4/97.5 | 84.5/91.8/96.4/97.4 | 71.5/85.0/92.5/94.3 |
| e06 strong regularizer | 84.4/91.7/96.5/97.3 | 84.4/91.7/96.5/97.3 | 84.4/91.7/96.5/97.4 | 77.0/87.7/94.0/95.6 |
| e07 guided .75 sigmoid2 | 84.6/91.8/96.2/97.3 | 84.6/91.8/96.2/97.3 | 84.6/91.8/96.3/97.3 | 77.9/88.5/94.5/96.1 |

Detaching the detail path is clearly harmful. Detaching only the prompt can
improve R1, but guided `.75` is not a stable mAP winner across seeds. The final
claim should remain “auxiliary local supervision”, not “guided mining is the
essential mechanism”.

### 5.11 FCU hierarchy and local supervision

Stage3-FCU x local 2x2, with Stage2 FCU always enabled and local excluded from
the main descriptor:

| Dataset/seed | Stage3 | Local train | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|---:|
| Duke/42 | off | off | 84.6 | 91.2 | 96.1 | 97.3 |
| Duke/42 | on | off | 84.3 | 91.5 | 96.1 | 97.3 |
| Duke/42 | off | on | 84.5 | 92.2 | 95.9 | 97.2 |
| Duke/42 | on | on | 84.7 | 92.1 | 96.5 | 97.4 |
| Market/42 | off | off | 90.7 | 95.5 | 98.6 | 99.1 |
| Market/42 | on | off | 90.1 | 95.4 | 98.5 | 99.0 |
| Market/42 | off | on | 90.9 | 95.8 | 98.5 | 99.3 |
| Market/42 | on | on | 90.5 | 95.4 | 98.6 | 99.2 |

Two-seed Stage3 effect with local on:

| Dataset | Seed | Stage3 on-off mAP | Stage3 on-off R1 |
|---|---:|---:|---:|
| Duke | 42 | +0.2 | -0.1 |
| Duke | 3407 | +0.1 | +0.1 |
| Market | 42 | -0.4 | -0.4 |
| Market | 3407 | -0.1 | -0.2 |

Duke Stage1 audit8:

| Experiment | FCU stages | Exchange | Local | Seed | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| e00 no exchange | [2] label only | no | on | 42 | 84.4 | 91.8 | 96.0 | 96.9 |
| e01 Stage1 | [1] | yes | on | 42 | 84.6 | 91.9 | 96.5 | 97.3 |
| e02 Stage2 | [2] | yes | on | 42 | 84.7 | 91.6 | 96.0 | 97.4 |
| e03 Stage1+2 | [1,2] | yes | on | 42 | **84.9** | 91.7 | 96.6 | 97.6 |
| e04 Stage1+2 | [1,2] | yes | on | 3407 | **84.8** | 92.1 | 96.3 | 97.3 |
| e05 Stage2 local-off | [2] | yes | off | 3407 | 84.5 | 91.5 | 96.1 | 97.0 |
| e06 A00 replica A | [2,3] | yes | on | 42 | 84.7 | 92.4 | 96.3 | 97.2 |
| e07 A00 replica B | [2,3] | yes | on | 42 | 84.2 | 91.7 | 96.4 | 97.2 |

At seed42, Stage1 contributes `+0.2/+0.1` whether Stage2 is off or on;
Stage2 contributes `+0.3/-0.2`, and the Stage1xStage2 interaction is exactly
zero at reported precision. Stage1+2 is the Duke candidate, while Stage3 and
the historical A00 `84.9/92.3` are not reliable cross-run winners.

### 5.12 PADE/PAM-style heterogeneous dual-view 28

File: `scripts/run_duke_fdmf_dual_view28.sh`. Results are from the server output
returned in this thread; no local summary copy exists.

| Experiment | Routing/ablation | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|
| a00 legacy best | legacy control | 84.4 | 91.7 | 96.2 | 97.4 |
| a01 shared p.50 | shared mask | 84.3 | 91.3 | 95.9 | 97.1 |
| a02 clean100 | no erasing | 79.4 | 89.9 | 94.9 | 96.3 |
| a03 shared p.25 | shared mask | 83.6 | 91.2 | 96.1 | 97.3 |
| a04 shared p.75 | shared mask | **84.6** | 91.5 | 96.1 | 97.2 |
| a05 diffmask p.50 | different masks | 82.9 | 90.9 | 95.8 | 96.9 |
| a06 independent p.50 | independent states | 82.6 | 91.0 | 95.5 | 96.9 |
| a07 anticorr random | anti-correlated | 82.0 | 90.6 | 95.7 | 96.8 |
| a08 anticorr PID22 | PID-balanced anti-correlated | 82.0 | 90.9 | 95.8 | 96.7 |
| a09 fixed CE | OSNet erased | 80.8 | 90.3 | 95.0 | 96.3 |
| a10 fixed EC | Mamba erased | 80.1 | 89.8 | 94.9 | 96.3 |
| a11 anchor CE p.50 | OSNet erased anchor | 80.6 | 90.3 | 95.0 | 96.2 |
| a12 anchor EC p.50 | Mamba erased anchor | 80.2 | 90.5 | 95.1 | 96.3 |
| a13 state sample | sampled branch state | 81.8 | 90.3 | 95.1 | 96.5 |
| b00 shared no-exchange | FCU off | 84.2 | 91.4 | 96.2 | 97.1 |
| b01 route no-exchange | diffmask, FCU off | 83.9 | 91.5 | 96.0 | 96.9 |
| b02 shared no-local | local off | 84.1 | 91.7 | 96.0 | 97.4 |
| b03 route no-local | diffmask, local off | 82.5 | 90.7 | 95.2 | 96.5 |
| b04 shared FDMF bypass | direct branches | 81.0 | 90.4 | 94.6 | 95.8 |
| b05 route FDMF bypass | diffmask, direct branches | 75.5 | 88.1 | 92.9 | 94.6 |
| b06 shared crop .25 | shared crop | 82.7 | 90.4 | 95.8 | 97.1 |
| b07 route crop .25 | diffmask plus crop | 82.2 | 90.0 | 95.5 | 96.8 |
| c00 best direction p.25 | OSNet-erased anchor | 80.8 | 90.3 | 95.3 | 96.3 |
| c01 shared dose .125 | shared erasing | 82.7 | 91.0 | 95.6 | 97.2 |
| c02 best direction p.75 | OSNet-erased anchor | 80.6 | 89.9 | 95.0 | 96.3 |
| c03 shared dose .375 | shared erasing | 84.0 | 91.4 | 95.9 | 97.0 |
| c04 reverse p.25 | Mamba-erased anchor | 80.1 | 89.7 | 95.0 | 96.3 |
| c05 reverse p.75 | Mamba-erased anchor | 80.2 | 89.9 | 95.1 | 96.2 |

This experiment rejects the simple “different image version per backbone”
hypothesis. Shared aligned erasing is consistently stronger. From PAM the
project borrowed clean/corrupted-view contrast; from ACC it borrowed role
differentiation and protected-anchor thinking. Neither borrowing validates a
new final module.

### 5.13 Protected-view 8 and feature-level role specialization 32

Protected-view 8 is a historical server result; its implementation was removed:

| Experiment | Protected design | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|
| e00 shared p.50 | control | 84.6 | 91.9 | 95.9 | 97.1 |
| e01 shared p.75 | control | 84.6 | **92.1** | 96.1 | 97.3 |
| e02 center-noise Mamba | protected Mamba | 84.4 | 92.1 | 96.1 | 97.3 |
| e03 center-noise OSNet | protected OSNet | 84.3 | 91.7 | 96.1 | 97.2 |
| e04 erase Mamba | protected Mamba | **84.7** | 91.6 | 96.6 | 97.4 |
| e05 erase OSNet | protected OSNet | 84.5 | 91.7 | 96.2 | 97.4 |
| e06 shared p.75 no-exchange | FCU off | 84.4 | 91.5 | 96.4 | 97.3 |
| e07 center OSNet no-exchange | FCU off | 84.3 | 91.5 | 96.2 | 97.3 |

The best protected result is only `+0.1 mAP/-0.5 R1` versus shared p.75 and is
not a reliable improvement.

The later role-specialization32 keeps the same input for both branches and
corrupts a selected feature map around FCU during training. Full main results
are in `logs/summary42.txt`:

| Experiment | Main intervention | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|
| a00 legacy | control | 84.5 | 91.6 | 96.3 | 97.4 |
| a01 random p.25 | OSNet pre-FCU | **84.6** | **92.1** | 96.3 | 97.3 |
| a02 random r.125 | drop dose | 84.6 | 91.7 | 96.3 | 97.3 |
| a03 random r.250 | drop dose | 84.4 | 91.9 | 96.3 | 97.3 |
| a04 random r.375 | drop dose | 84.5 | 92.1 | 96.2 | 97.1 |
| a05 random r.500 | drop dose | 84.2 | 91.6 | 96.2 | 97.1 |
| a06 random p.75 | probability | 84.3 | 91.5 | 96.3 | 97.2 |
| a07 random p1.0 | probability | 84.5 | 91.8 | 96.1 | 97.1 |
| b00 batch p.50 | mask type | 84.3 | 91.7 | 96.3 | 97.3 |
| b01 batch p1.0 | mask type | 84.5 | 91.8 | 96.4 | 97.5 |
| b02 stripe p.50 | mask type | 84.5 | 92.1 | 96.3 | 97.2 |
| b03 attention p.50 | mask type | 84.1 | 91.7 | 96.2 | 97.1 |
| b04 batch mean-fill | fill | 84.6 | 91.9 | **96.6** | 97.2 |
| b05 random mean-fill | fill | 84.3 | 91.9 | 95.9 | 97.4 |
| b06 post-FCU random | location | 84.1 | 91.8 | 96.3 | 97.1 |
| b07 attention r.125 | mask type | 84.5 | 91.9 | 96.5 | 97.3 |
| c00 compensation .01 | cosine | 84.6 | 91.8 | 96.1 | 97.3 |
| c01 compensation .025 | cosine | 84.6 | 91.9 | 96.5 | 97.3 |
| c02 compensation .05 | cosine | 84.6 | 91.9 | 96.4 | 97.4 |
| c03 compensation .10 | cosine | 84.5 | 91.8 | 96.1 | 97.4 |
| c04 start20/ramp20 | schedule | 84.5 | 91.7 | 96.0 | 97.5 |
| c05 start40/ramp20 | schedule | 84.5 | 91.7 | 96.4 | 97.2 |
| c06 target gradient | detach target off | 84.4 | 91.9 | 96.3 | 97.4 |
| c07 detach source | detach both | 84.3 | 91.3 | 96.1 | 97.1 |
| d00 no-local control | local off | 84.3 | 91.3 | 96.1 | 97.3 |
| d01 no-local mask | local off | 84.4 | 91.4 | 96.3 | 97.1 |
| d02 no-local compensation | local off | 84.5 | 92.1 | 96.3 | 97.2 |
| d03 completed no-comp | local route | 84.4 | 92.2 | 96.2 | 97.4 |
| d04 masked compensation | local route | 84.5 | 91.8 | 96.3 | 97.4 |
| d05 completed compensation | local route | 84.4 | 91.8 | 96.4 | 97.5 |
| d06 random completed | local route | 84.4 | 92.2 | 96.2 | 97.4 |
| d07 attention completed | local route | 84.2 | 91.2 | 96.3 | 97.4 |

Light pre-FCU feature corruption produces at most a seed-scale `+0.1 mAP`.
Compensation losses, schedules, attention masks and local routing do not exceed
the simple control convincingly. Keep this mechanism out of the final network.

### 5.14 Shared-geometry appearance augmentation 28

Files: `logs/summary_aug28.txt` and the corresponding homologous augmentation
script. Geometry is shared/aligned; only appearance may differ.

| Experiment | Appearance target/setting | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|
| a00 current p.75 | shared erase control | **84.9** | **92.3** | 96.1 | 97.3 |
| a01 shared color | p.50 s.20 | 84.0 | 91.1 | 96.2 | 97.2 |
| a02 shared gray | p.50 s.50 | 84.5 | 91.9 | 96.1 | 97.2 |
| a03 shared blur | p.50 s1.0 | 84.5 | 92.0 | 96.1 | 97.4 |
| b00 color Mamba | p.25 s.20 | 84.5 | 91.7 | 96.0 | 97.1 |
| b01 color Mamba | p.50 s.20 | 84.6 | 91.6 | 96.1 | 97.3 |
| b02 color Mamba | p.75 s.20 | 84.7 | 91.5 | 96.3 | 97.6 |
| b03 color OSNet | p.50 s.20 | 84.4 | 91.6 | 96.2 | 97.3 |
| b04 color random branch | p.50 s.20 | 84.3 | 91.9 | 96.4 | 97.4 |
| c00 gray Mamba | p.25 s.50 | 84.5 | 91.8 | 96.3 | 97.4 |
| c01 gray Mamba | p.50 s.50 | 84.4 | 92.1 | 96.2 | 97.3 |
| c02 gray Mamba | p.75 s.50 | 84.7 | 92.1 | 96.2 | 97.2 |
| c03 gray OSNet | p.50 s.50 | 84.5 | 92.0 | 96.3 | 97.1 |
| c04 gray random branch | p.50 s.50 | 84.3 | 91.4 | 96.2 | 97.1 |
| d00 blur Mamba | p.25 s1.0 | 84.7 | 91.9 | 96.1 | 97.1 |
| d01 blur Mamba | p.50 s1.0 | 84.7 | 91.7 | 95.9 | 97.3 |
| d02 blur Mamba | p.75 s1.0 | 84.4 | 91.7 | 96.1 | 97.4 |
| d03 blur OSNet | p.50 s1.0 | **84.8** | 91.8 | 96.2 | 97.4 |
| d04 blur random branch | p.50 s1.0 | 84.7 | 91.6 | 96.2 | 97.3 |
| e00 color Mamba | p.50 s.10 | 84.7 | 91.9 | 96.1 | 97.4 |
| e01 color Mamba | p.50 s.40 | 84.6 | 92.0 | 95.9 | 97.5 |
| e02 gray Mamba | p.50 s.25 | 84.6 | 92.1 | 96.0 | 97.5 |
| e03 gray Mamba | p.50 s.75 | 84.3 | 91.8 | 96.2 | 97.1 |
| e04 blur Mamba | p.50 s.50 | 84.5 | 91.7 | 96.3 | 97.4 |
| e05 blur Mamba | p.50 s1.50 | 84.6 | 91.7 | 96.1 | 97.1 |
| f00 color Mamba, no erase | p.50 s.20 | 79.1 | 89.4 | 94.4 | 95.7 |
| f01 gray Mamba, no erase | p.50 s.50 | 79.2 | 89.7 | 94.7 | 96.0 |
| f02 blur Mamba, no erase | p.50 s1.0 | 79.4 | 89.7 | 94.7 | 96.0 |

The apparent control `84.9/92.3` is anomalously high. Exact later replicas are
`84.7/92.4` and `84.2/91.7`; mean `84.45/92.05`, spread `0.50/0.70`.
Therefore the table does not prove that every appearance augmentation is
strictly harmful, but it provides no reproducible improvement either. Shared
random erasing is essential: removing it drops mAP to about `79.1-79.4`.

### 5.15 MSMT17 FDMF role/FCC 8 and Duke component6

The MSMT17 results below were returned from the server. The abandoned role/FCC
implementation and script were subsequently deleted, so this is a historical
result table rather than an active reproducible path:

| Experiment | Variant | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|
| e00 legacy | legacy Mamba | 68.5 | 85.6 | 92.3 | 94.0 |
| e01 no Mamba | depth0 | **68.7** | **85.8** | **92.7** | **94.3** |
| e02 FCC gate | feature cross correction | 68.4 | 85.7 | 92.2 | 94.2 |
| e03 role Mamba-SSM | role mixer | 68.5 | 85.1 | 92.0 | 94.0 |
| e04 role OSNet-SSM | reversed role | 68.6 | 85.7 | 91.9 | 93.9 |
| e05 output gate | role plus gate | 68.6 | 85.4 | 92.4 | 94.3 |
| e06 relation BC | role plus relation BC | 68.6 | 85.5 | 92.2 | 94.1 |
| e07 role full | relation plus gate | 68.6 | 85.8 | 92.1 | 94.0 |

No-Mamba wins this MSMT17 run. FCC/role gives no useful mAP gain and the
supposed “correct” role assignment is worse than the reverse. This does not by
itself prove that Mamba is useless on Duke/Market, because earlier Duke
checkpoints showed the complete FDMF descriptor has stable conditional value:

| Seed | FDMF-only mAP/R1 | Mamba+OSNet | Mamba+FDMF+OSNet | Conditional FDMF gain |
|---:|---:|---:|---:|---:|
| 42 | 81.66/91.07 | 82.40/91.43 | 84.08/91.52 | +1.155/+0.180 |
| 3407 | 81.14/90.44 | 82.19/90.71 | 83.79/91.11 | +1.193/+0.494 |

The running Duke component6 is the formal causal audit:

| Experiment | Projection | Mamba mixer | Outer MLP | MSEF | Status |
|---|---:|---:|---:|---:|---|
| fc6 e00 direct concat | bypassed | n/a | n/a | n/a | running/pending |
| fc6 e01 projection only | on | off | on | off | running/pending |
| fc6 e02 Mamba no-MSEF | on | on | on | off | running/pending |
| fc6 e03 MSEF-only | on | off | on | on | running/pending |
| fc6 e04 Mamba+MSEF | on | on | on | on | running/pending |
| fc6 e05 mixer-only | on | on | off | off | running/pending |

Fixed controls are Duke seed42, Stage1+2 FCU, semantic-detail local training,
local inference weight `0`, shared erasing `p=.75`, and raster scan. All map
fusion groups construct the same modules and use forward switches, preserving
the initialization RNG trajectory of later heads. Do not fill this table with
assumed values; wait for the server summary.

### 5.16 Earlier FDMF evolution results

The earlier fusion handoff records why the project moved away from frequency
decomposition toward same-scale map fusion:

| Experiment | mAP | R1 | Interpretation |
|---|---:|---:|---|
| raw frequency FDMF, no Mamba | 82.1 | 90.0 | weak frequency baseline |
| frequency FDMF plus Mamba | 83.4 | 90.8 | Mamba recovered part of the loss |
| plain same-scale map-Mamba | 83.7 | 91.2 | frequency split unnecessary |
| map-Mamba plus MSEF | 83.8 | 91.6 | small MSEF gain |
| map-Mamba plus scaled MSEF | 83.8 | 91.6 | residual scale no extra gain |

Early Duke FCU+FDMF direction search:

| FCU setting | mAP | R1 |
|---|---:|---:|
| Stage2 OSNet->Mamba | 83.5 | 90.9 |
| Stage2 Mamba->OSNet | 83.6 | 91.1 |
| Stage2 bidirectional | 83.4 | 90.9 |
| Stage3 OSNet->Mamba | 83.6 | 91.1 |
| Stage3 Mamba->OSNet | 83.7 | 91.1 |
| Stage3 bidirectional | 83.6 | 91.2 |
| Stage2+3 bidirectional | 83.7 | 90.8 |
| Stage2 O->M + Stage3 M->O | **83.9** | 91.4 |
| Same weights, three-descriptor inference | **84.0** | **91.5** |

These older absolute numbers should not be compared directly to the latest
augmentation/local hierarchy runs because optimizer, descriptor and training
controls changed. They establish the architectural lineage only.

### 5.17 Consolidated experiment decision table

| Question | Evidence | Decision |
|---|---|---|
| Does local help? | Separate local checkpoints gain about .2-.5 mAP | Keep auxiliary local supervision |
| Does local descriptor help? | Same checkpoint direct gain 0-.1 mAP | Exclude it from final descriptor |
| Are part/confidence heads useful? | local24 e16-e21 do not beat simple loss | Do not use |
| Is guided .75 essential? | seed42 positive, seed3407 not reproduced | Optional training detail, weak claim |
| Is Stage3 FCU useful? | Duke tiny positive, Market negative across two seeds | Remove from unified structure |
| Is Stage1 FCU useful? | Duke +.2 mAP at seed42 and positive second-seed margin | Duke candidate; validate cross-dataset |
| Should branches receive different images? | dual-view28 consistently worse | No |
| Does feature corruption create roles? | role32 at most +.1 mAP | No final module |
| Do appearance-asymmetric views help? | aug28 no reproducible winner | No |
| Should scan be dynamic/four-way? | Duke tiny gain, Market zero | Keep raster |
| Are FDMF Mamba/MSEF both necessary? | MSMT result questions Mamba | Await component6 |

## 6. Numerical Stability and Known Failures

### 6.1 Triplet stability

`loss/triplet_loss.py` now performs normalization, Euclidean distance, cosine
distance, and pairwise accumulation in FP32. The soft-margin branch uses stable
`F.softplus` behavior instead of an overflow-prone backend path.

This should not reduce meaningful performance. It computes the same objective
more accurately under AMP and prevents FP16 squared-distance overflow.

### 6.2 Non-finite loss guard

`processor/processor.py` checks the loss before backward/optimizer step. On a
non-finite loss it reports:

- bad loss terms,
- tensor finite/NaN/Inf counts,
- maximum absolute values,
- non-finite parameter names,
- OSBBM activity.

It then terminates the experiment before writing corrupt parameters. This does
not alter successful optimizer steps; it only stops a failed run.

### 6.3 Market crash

A Market run crashed at epoch 70 with:

```text
tri_mamba = inf
tri_fdmf = inf
all input, score, feature, and parameter tensors finite
OSBBM_active = False
```

This identified triplet pairwise-distance/soft-margin overflow rather than an
OSBBM-specific fault. Retry under the FP32 triplet fix is required when needed.

### 6.4 Configuration-key mismatch

Market initially failed YACS merge due to unsupported keys such as:

- `FDMF_FILTER_TYPE`
- `FDMF_STRIPE_DEPTH`
- `STAGE4_STRIPE_LOCAL_ENABLED`

They were removed from the dataset YAML files because the current defaults/code
do not define or consume them. Removing dead unsupported keys does not change
the active baseline behavior.

### 6.5 Loss API mismatch

`processor.py` began passing `epoch=epoch` to the loss function, causing:

```text
TypeError: loss_func() got an unexpected keyword argument 'epoch'
```

The loss API was subsequently aligned. Preserve this compatibility when editing
`make_loss.py` or `processor.py`.

## 7. What Has Been Rejected or Deprioritized

Do not repeat these broad searches without a new mechanism-level reason:

- Hard complete decorrelation of Mamba/FDMF/OSNet features.
- `shared_m` and `shared_mo` residual-removal complementarity.
- Joint descriptor supervision at the tested high weights.
- Branch Drop as a cure for the above collapse.
- Raw covariance penalties.
- Peer complementarity losses in their tested form.
- RATR as the main solution.
- Strong semantic positional prior (`2.0`).
- Treating any positive local inference weight as a major source of gain; the
  measured direct contribution is only `0-0.1 mAP`.
- Replacing default raster scan with four-way scan based on current evidence.
- OSBBM as part of the formal baseline.
- Separate images, different random-erasing masks, anti-correlated views, or
  fixed clean/erased routing for MambaVision and OSNet.
- Protected-view center noise and feature-space compensation losses.
- Broad color/grayscale/blur asymmetry as a replacement for shared erasing.
- Stage3 FCU as a universal cross-dataset improvement.
- FDMF role-conditioned mixer and FCC gate; their code has been removed.
- Part-specific local ID, part-average triplet, and confidence weighting in the
  tested forms.

The conceptual lesson is important: useful branches should share identity
semantics while contributing different errors/details. Complementarity is not
the same as forced orthogonality.

## 8. Next Experiments

### Priority 1: finish and interpret Duke component6

This is the blocking structural experiment. Use the following decision rules:

- If projection-only is not worse than full Mamba+MSEF, simplify FDMF.
- If Mamba-no-MSEF beats projection-only, Mamba has independent value.
- If MSEF-only beats projection-only, MSEF has independent value.
- Only claim synergy if the Mamba x MSEF factorial interaction is positive.
- Use mixer-only to determine whether the outer MLP is necessary.

### Priority 2: validate Stage1 outside Duke

Run Stage2 versus Stage1+2 with identical local supervision and local inference
weight `0` on Market and preferably MSMT17. Do not add Stage3. Two seeds are
not required for the first cross-dataset screen; reproduce only if the Stage1
margin is meaningful.

### Priority 3: make FDMF match its name

Current `SameScaleMambaFusion` performs projected concatenation followed by
Mamba/MSEF; it does not explicitly compute a feature difference. If component6
confirms that map fusion is useful, test a residual-safe difference-conditioned
fusion:

```text
D = [abs(norm(M)-norm(O)), norm(M)*norm(O)]
F0 = Conv1x1([M,O])
F  = F0 + alpha * Conv1x1(gate(D) * (norm(M)-norm(O)))
```

Initialize `alpha` at zero or a small value and preserve the concat baseline.
Do not use a pure convex gate `G*M+(1-G)*O`, which can discard complementary
branch information.

### Priority 4: clean paper ablation and efficiency

Required final sequence:

```text
MambaVision only
OSNet only
parallel direct descriptor concat
+ Stage2 FCU
+ Stage1 FCU (if cross-dataset validated)
+ projected map fusion
+ Mamba refinement (if component6 validates it)
+ MSEF (if component6 validates it)
+ semantic-detail local supervision, descriptor excluded
```

Report parameters, FLOPs, training cost, inference latency, descriptor
dimension, two-seed Duke results, and at least Market/MSMT17 results. Use the
existing CKA/error-overlap evidence as mechanism support.

### Explicitly out of scope

- OCC-Duke/OSBBM is not a current implementation priority.
- Do not reopen PADE/PAM-style dual views, protected views, role specialization,
  broad appearance augmentation, RATR, or dynamic scan without a new causal
  hypothesis.

## 9. Paper Positioning

The network can support a paper if the contribution is framed as:

> Stage-aware heterogeneous interaction and difference-modulated fusion between
> a global state-space branch and an omni-scale convolutional detail branch,
> with semantic-guided local auxiliary supervision.

Do not frame it simply as "MambaVision + OSNet + local features". CNN/global
model dual branches already exist. The defensible novelty candidates are:

1. Early/mid-stage heterogeneous FCU interaction, currently Stage1+2 on Duke.
2. Heterogeneous feature-map fusion with Mamba/MSEF components only if the
   running component6 proves their necessity.
3. Semantic-guided pooling of OSNet detail using Mamba semantic/foreground maps
   as a training-time auxiliary branch.

Critical naming warning: the current FDMF implementation does not yet perform
explicit difference modulation; it performs aligned projection, concatenation,
Mamba refinement and MSEF. Either implement and validate the residual-safe
difference mechanism in Priority 3, or rename the block before submission.
Do not claim “difference-modulated” based only on concatenation.

Current publication assessment:

- Enough for a Chinese domain journal after clean ablation and provenance audit.
- Potentially suitable for TMM/TCSVT/Pattern Recognition style submission with
  explicit difference conditioning, stronger mechanism evidence, complexity
  analysis, and cross-dataset Stage1 validation.
- Not yet strong enough for a top vision main conference based only on current
  gains; novelty must be distinguished from existing CNN+Transformer/Mamba
  fusion methods.

## 10. Useful Commands and Files

Current formal configs:

- `configs/DukeMTMC/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml`
- `configs/Market/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml`
- `configs/MSMT17/mambavision_tiny_osnet_fdmf_msef_stage_fcu_b64k4.yml`

Main code:

- `model/make_model.py`: FCU, FDMF, scan modes, semantic-detail local branch.
- `loss/make_loss.py`: branch ID/triplet weights and auxiliary losses.
- `loss/triplet_loss.py`: FP32 triplet distance stability.
- `processor/processor.py`: training, OSBBM application, logging, non-finite guard.
- `config/defaults.py`: all optional keys and defaults.
- `solver/make_optimizer.py`: OSNet and fusion LR/WD parameter groups.
- `utils/osbbm.py`: OSBBM augmentation variants.
- `utils/feature_complementarity.py`: branch analysis and weight sweeps.

Experiment summaries:

- `logs/summary/summary_all.txt`: Complementarity-24.
- `logs/summary.txt`: peer complementarity.
- `logs/summary20.txt`: optimizer/RATR/local/OSBBM hypotheses.
- `logs/summary_s42.txt`: older broad stripe-local24.
- `logs/summary_shortlist_2seed.txt`: two-seed local shortlist.
- `logs/summarydetail4.txt`: first semantic-detail tests.
- `logs/summary_sdetail20.txt`: semantic-detail structure search.
- `logs/summary42.txt`: feature-level role-specialization32 and descriptor decomposition.
- `logs/summary_aug28.txt`: shared-geometry appearance augmentation28.
- `logs/summary_fcu_local.txt`: Market Stage3-FCU x local seed42.

Server-only/user-returned summaries not currently present in this worktree:

- new `l24_e00-e23` local24 and inference decomposition;
- Guided6 and Causal8;
- dual-view28 and protected-view8;
- Stage1 audit8 and the two-seed Stage3 final comparison;
- MSMT17 role/FCC8 historical results;
- Duke component6, which is still running at this update.

Do not confuse `logs/summary_s42.txt` with the new local24. It belongs to the
older broad stripe-local experiment.

Scan command examples:

```bash
# Four Duke scan variants
bash scripts/run_duke_fdmf_scan4.sh 0,1 4

# Four Duke + four Market variants
bash scripts/run_fdmf_scan8_duke_market.sh 0,1 4
```

General experiment convention:

- `GPU_IDS` is a comma-separated allowed GPU list.
- `MAX_JOBS` is the total number of simultaneous processes.
- Existing pool scripts assign jobs round-robin; they do not dynamically query
  GPU utilization.
- `SKIP_COMPLETED=1` skips final checkpoints/logs already present.

## 11. First Actions for the Next Window

1. Read this file, then inspect `git status --short`.
2. Retrieve the completed `run_duke_fdmf_component6.sh` summary from the server.
3. Interpret projection/Mamba/MLP/MSEF effects before changing FDMF again.
4. If Mamba/MSEF is justified, design the explicit residual-safe difference
   modulation; otherwise simplify/rename FDMF.
5. Validate Stage1+2 versus Stage2-only on Market/MSMT17.
6. Keep local supervision on and local inference weight at zero for mechanism
   and hierarchy comparisons.
7. Do not reopen RATR, hard decorrelation, PADE-style dual views, broad
   augmentation, role/FCC, OSBBM, or dynamic scan without new evidence.

## 12. Superseding Update: SPA-FDMF (2026-07-26)

This section supersedes the older FDMF priorities above.

### Evidence that closed the previous fusion directions

- Duke component6 confirmed that projected map fusion is the stable carrier:
  projection-only reached `84.5/91.8`; Mamba without MSEF reached
  `84.8/92.4`; Mamba+MSEF reached `84.8/92.2` under the weighted descriptor.
- HR-SSF did not beat its legacy control (`84.7/91.5`).
- Dual-stream scan/JamMa-style variants did not beat the legacy projected
  FDMF (`84.8/92.0` weighted); the direct dual descriptor was consistently
  weaker.
- CASM and its loss/readout search did not establish a stable improvement;
  the returned default weighted results were around `84.3`, below the
  established FDMF range. Shuffled/correct-pair controls also failed to show
  the required causal ordering.

Therefore the code paths for HR-SSF, SRC-JS/JamMa and CASM, their configuration
keys, auxiliary loss, and dedicated scripts were removed. This cleanup does
not remove FCU, semantic-detail local supervision, the stable FDMF path, or
the local head's independent part-ID options.

### New hypothesis: Shared Prototype Alignment before FDMF

The new implementation is inspired by AAformer's learned PART tokens and
optimal-transport grouping, but adapts the idea to same-image heterogeneous
dual backbones:

```text
MambaVision map M ---- branch query ----\
                                        shared K prototypes
OSNet map O --------- branch query ----/       |
                                                +-- Softmax/Sinkhorn assignments
M receives aligned OSNet region tokens <-------+
O receives aligned Mamba region tokens <-------+
                    |
Proj1x1([M_aligned, O_aligned]) -> Mamba -> MSEF -> GeM
```

- The two branches retain separate query projections but use the same
  prototype indices as a semantic coordinate system.
- The existing Stage3 semantic-detail masks can be detached and used as a
  `K=2` assignment prior. They are not used as an inference descriptor.
- Sinkhorn assignments impose balanced region usage; Softmax is the necessary
  unbalanced control.
- Cross-branch region tokens are broadcast back through residual scales
  initialized at `0.1`, so the original backbone maps remain explicit.
- The stable `1x1 projection -> one SpatialMambaBlock -> MSEF` path is retained
  after alignment. This experiment tests alignment, not another carrier.
- Default inference remains normalized MambaVision + FDMF + OSNet. Additional
  `aligned_pair` and `aligned_pair_fdmf` modes are diagnostic only.

Implementation/configuration:

- `model/make_model.py`: `SharedPrototypeAlignment` and simplified
  `SameScaleMambaFusion`.
- `config/defaults.py`: `FDMF_ALIGNMENT_*` keys; alignment is disabled by
  default so historical baseline behavior is retained.
- `scripts/run_duke_fdmf_spa8.sh`: eight Duke experiments with two-GPU,
  four-worker fixed-slot scheduling.

SPA8 causal design:

1. `e00`: exact stable FDMF control, no alignment.
2. `e01`: shared prototypes + Softmax, no FDMF Mamba scan.
3. `e02`: shared prototypes + Sinkhorn, no FDMF Mamba scan.
4. `e03`: proposed Sinkhorn alignment + stable FDMF Mamba.
5. `e04`: remove the existing local-mask prior.
6. `e05`: replace shared prototypes with independent branch prototypes.
7. `e06`: shuffle peer region tokens across samples during training.
8. `e07`: change both local regions and alignment prototypes from `K=2` to
   `K=4`.

Primary decision criterion: `e03` must stay in the established `84.5+` range,
beat `e00` by a meaningful margin, and outperform `e06`. If correct-pair
alignment is not better than shuffled alignment, do not claim semantic
correspondence and stop this line.

Server command:

```bash
bash scripts/run_duke_fdmf_spa8.sh 0,1 4
```

No local Python/model execution was performed for this update. The code was
reviewed statically; final import, forward, CUDA and shell syntax checks must be
performed on the server before launching the full run.

### SPA numerical-stability correction

The first server run showed that every alignment-enabled group developed a
non-finite fused descriptor around epoch 14, while the legacy group and both
backbones remained finite. The finite portion of the FDMF descriptor had
already reached an absolute magnitude near `493.7`; parameters were still
finite. This identified AMP forward overflow caused by an unbounded
cross-region residual, not a loss-function failure.

The alignment path was corrected by keeping assignment/cross-token operations
in FP32, applying parameter-free LayerNorm before and after each cross-branch
linear transform, clamping each residual scale to `[-0.25, 0.25]`, and assigning
alignment parameters a separate `1x` base learning rate. The established FDMF
projection/Mamba remains at `3x`. Do not use checkpoints from the unstable
pre-correction SPA run.

## 13. Superseding Update: ODSMF20 (2026-07-26)

SPA was numerically repaired but did not justify further development. Its first
four completed Duke results were `84.6/91.7` (legacy), `83.6/91.5` (Softmax),
`83.4/91.6` (Sinkhorn) and `84.1/91.7` (Sinkhorn+Mamba). The alignment code,
configuration keys, optimizer exception and dedicated SPA script were therefore
removed. The historical `84.6/91.7` result remains the external reference.

The next hypothesis is Orthogonal Dual-State Mamba Fusion (ODSMF). After
branch-specific projection and normalization it forms

```text
C = (M + O) / sqrt(2)       shared identity evidence
D = (M - O) / sqrt(2)       signed heterogeneous discrepancy
```

`C` and `D` are modeled by role-specific paths that share the core Mamba mixer
by default. A spatial/channel reliability gate modulates only the discrepancy
correction. The stable output is

```text
B = Proj1x1([C, D])
F = B + alpha_C * P_C(C' - C) + alpha_D * G * P_D(D' - D)
F -> MSEF -> GeM
```

Both residual scales start at `0.1` and are bounded to `[-0.5, 0.5]`. Learned
gates start exactly at one, and the multiplicative gate predictor runs in FP32.
Default inference remains normalized MambaVision + FDMF + OSNet with weights
`1.0/0.75/0.4`; consensus/discrepancy descriptors are diagnostics only.

`scripts/run_duke_fdmf_odsmf20.sh` contains 20 new-module internal ablations:
state necessity, mixer/norm sharing, width, depth, packed scan, five bases,
four gates, correct-pair shuffling, discrepancy gradient detachment, carrier
removal and zero residual initialization. It deliberately does not spend a
training slot repeating the extensively validated legacy baseline.

Server command:

```bash
bash scripts/run_duke_fdmf_odsmf20.sh 0,1 4
```

No local Python/model execution was performed. `git diff --check` passed. The
local WSL launcher was unavailable, so `bash -n`, import and CUDA forward checks
must be run on the server before committing the full sweep.

## 14. ODSMF20 Result and Pure-State Follow-up (2026-07-27)

ODSMF20 did not exceed the historical projection-Mamba range. The complete
carrier-based anchor reached `84.3/91.6` under the weighted descriptor. The two
best weighted configurations were discrepancy-only (`84.5/91.9`) and the
carrier-free direct dual-state readout (`84.5/91.9`, R5/R10 `96.6/97.7`).

The load-bearing finding was contrary to the original residual-safe hypothesis:
removing the projection carrier improved `fdmf_only` from `78.5` to `79.7` and
the weighted descriptor from `84.3` to `84.5`. The carrier acted as a shortcut,
so ablations performed around the carrier anchor cannot be assumed to transfer
to the carrier-free winner. The complete dual-state anchor also did not beat
its consensus-only or discrepancy-only controls. Do not claim established
orthogonal dual-state synergy from ODSMF20.

The new `purestate8` suite reuses the completed `od20_e18` result as its
unretrained anchor, holds `carrier=False` in every new group and tests only:

1. no Mamba scan (`depth=0`);
2. consensus-only scan;
3. discrepancy-only scan;
4. packed C/D scan;
5. channel gate instead of separable gate;
6. no reliability gate;
7. shuffled discrepancy causal control;
8. detached discrepancy gradient control.

The common ODSMF driver now supports `depth=0` and both experiment suites. The
dedicated server command is:

```bash
bash scripts/run_duke_fdmf_purestate8.sh 0,1 4
```

The original 20-group command remains backward compatible. No local Python,
model construction or training was run for this update.
