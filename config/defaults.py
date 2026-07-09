from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Pooling type: 'gem', 'avg', 'max', 'avg_max' (消融实验用)
_C.MODEL.POOLING_TYPE = 'gem'

# Local stripe descriptors on the final feature map.
_C.MODEL.LOCAL_STRIPE = CN()
_C.MODEL.LOCAL_STRIPE.ENABLED = False
_C.MODEL.LOCAL_STRIPE.NUM_STRIPES = 4
_C.MODEL.LOCAL_STRIPE.LOSS_WEIGHT = 0.2
_C.MODEL.LOCAL_STRIPE.USE_TRIPLET = True
_C.MODEL.LOCAL_STRIPE.INFERENCE = 'concat'  # global, local, concat
_C.MODEL.LOCAL_STRIPE.POOLING_TYPE = 'gem'
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION = CN()
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.ENABLED = False
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.MIXER_TYPE = 'mambavision'  # mambavision, conv
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.TOKEN_POOLING_TYPE = 'gem'
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.MODE = 'even'  # head, tail, even
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.KERNEL_SIZE = 3
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.MLP_RATIO = 2.0
_C.MODEL.LOCAL_STRIPE.TOKEN_INSERTION.INIT_SCALE = 1e-3

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# MambaVision配置
_C.MODEL.MAMBAVISION = CN()
_C.MODEL.MAMBAVISION.GLOBAL_STAGES = []  # Stages using global attention (no window partition)
_C.MODEL.MAMBAVISION.SASF_STAGES = []    # Stages with StateFusion enabled
_C.MODEL.MAMBAVISION.USE_FINE_BRANCH = False # Enable OSNet-style fine-grained branch
_C.MODEL.MAMBAVISION.FINE_BRANCH_BLOCKS = 3  # Number of blocks in fine branch
_C.MODEL.MAMBAVISION.AUX_LOSS_WEIGHT = 0.5   # Weight for auxiliary branch loss (0.2~0.5 recommended)

# SFM (SimpleFusionMamba) 跨阶段特征融合配置
_C.MODEL.MAMBAVISION.USE_SFM = False      # 是否启用 SFM 模块
_C.MODEL.MAMBAVISION.SFM_NUM_LAYERS = 3   # 已升级为 3 层层级聚合 (Stage 1-4)
_C.MODEL.MAMBAVISION.SFM_DEPTHS = [1, 1, 1]  # [S1+2, F12+3, F23+4] 各级融合 Block 数
_C.MODEL.MAMBAVISION.SFM_DROP_PATH = 0.1  # SFM 模块内部的 DropPath 概率
_C.MODEL.MAMBAVISION.SFM_POOLING_TYPE = 'gem'  # SFM 分支池化类型 ('gem', 'avg', 'max', 'avg_max')

# Descriptor-level OSNet + MambaVision fusion.
_C.MODEL.OSNET_FUSION = CN()
_C.MODEL.OSNET_FUSION.ENABLED = False
_C.MODEL.OSNET_FUSION.OSNET_TYPE = 'osnet_x1_0'
_C.MODEL.OSNET_FUSION.PRETRAIN_PATH = ''
_C.MODEL.OSNET_FUSION.FREEZE_OSNET = False
_C.MODEL.OSNET_FUSION.OSNET_LOSS_WEIGHT = 0.5
_C.MODEL.OSNET_FUSION.FUSED_LOSS_WEIGHT = 1.0
_C.MODEL.OSNET_FUSION.FUSION_TYPE = 'descriptor'  # descriptor / stage_fcu / fdmf
_C.MODEL.OSNET_FUSION.FUSION_NORM = 'none'  # none / branch / weighted_branch; fdmf use none
_C.MODEL.OSNET_FUSION.FUSION_BETA = 1.0     # OSNet branch scale for weighted_branch
_C.MODEL.OSNET_FUSION.FDMF_COMPRESSED_CHANNELS = 64  # deprecated; kept for old configs/scripts
_C.MODEL.OSNET_FUSION.FDMF_LOWPASS_KERNEL = 5        # deprecated; kept for old configs/scripts
_C.MODEL.OSNET_FUSION.FDMF_HIGHPASS_KERNEL = 3       # deprecated; kept for old configs/scripts
_C.MODEL.OSNET_FUSION.FDMF_HAMMING_WINDOW = True     # deprecated; kept for old configs/scripts
_C.MODEL.OSNET_FUSION.FDMF_FILTER_TYPE = 'none'      # deprecated; frequency filtering has been removed
_C.MODEL.OSNET_FUSION.FDMF_FUSED_FORM = 'raw_fdmf'  # raw_fdmf / mamba_fdmf / fdmf_only
_C.MODEL.OSNET_FUSION.FDMF_MAMBA_DEPTH = 1
_C.MODEL.OSNET_FUSION.FDMF_MAMBA_D_STATE = 8
_C.MODEL.OSNET_FUSION.FDMF_MAMBA_D_CONV = 3
_C.MODEL.OSNET_FUSION.FDMF_MAMBA_INIT_SCALE = 0.1
_C.MODEL.OSNET_FUSION.FDMF_MAMBA_BIDIRECTIONAL = True
_C.MODEL.OSNET_FUSION.FDMF_MLP_RATIO = 2.0
_C.MODEL.OSNET_FUSION.FDMF_STRIPE_DEPTH = 0
_C.MODEL.OSNET_FUSION.FDMF_STRIPE_NUM = 4
_C.MODEL.OSNET_FUSION.FDMF_STRIPE_SHARE_PARAMS = True
_C.MODEL.OSNET_FUSION.FDMF_MSEF_ENABLED = True
_C.MODEL.OSNET_FUSION.FDMF_MSEF_REDUCTION_RATIO = 16
_C.MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_ENABLED = False
_C.MODEL.OSNET_FUSION.FDMF_MSEF_RES_SCALE_INIT = 0.1
_C.MODEL.OSNET_FUSION.FCU_ENABLED = False   # optionally combine stage_fcu exchange with fdmf
_C.MODEL.OSNET_FUSION.FCU_INIT_SCALE = 0.1  # residual scale for stage_fcu exchange
_C.MODEL.OSNET_FUSION.FCU_STAGES = [2, 3]   # stage_fcu exchange stages: any subset of [2, 3]
_C.MODEL.OSNET_FUSION.FCU_DIRECTION = 'bidirectional'  # bidirectional / osnet_to_mamba / mamba_to_osnet
_C.MODEL.OSNET_FUSION.FCU_STAGE2_DIRECTION = ''  # optional override; empty uses FCU_DIRECTION
_C.MODEL.OSNET_FUSION.FCU_STAGE3_DIRECTION = ''  # optional override; empty uses FCU_DIRECTION
_C.MODEL.OSNET_FUSION.FCU_GATE_TYPE = 'none'  # none / channel
_C.MODEL.OSNET_FUSION.FCU_GATE_REDUCTION = 16
_C.MODEL.OSNET_FUSION.FCU_GATE_INIT_BIAS = 0.0
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_ENABLED = False
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_NUM_STRIPES = 4
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_SHARE_PARAMS = True
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_DEPTH = 1
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_D_STATE = 8
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_D_CONV = 3
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_INIT_SCALE = 0.1
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MAMBA_BIDIRECTIONAL = True
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_MLP_RATIO = 2.0
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_OVERLAP_RATIO = 0.0
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONTEXT_MIXER_TYPE = 'none'  # none / dwconv / mlp
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONTEXT_MIXER_KERNEL = 3
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_CONTEXT_MIXER_INIT_SCALE = 1e-3
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_RELIABILITY_GATE_ENABLED = False
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_RELIABILITY_HIDDEN_RATIO = 0.25
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_RELIABILITY_INIT_BIAS = 2.0
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_DIM = 0  # 0 keeps the aligned stage channel dim
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LOSS_WEIGHT = 0.2
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_INFER_WEIGHT = 0.3
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_SUPERVISION_ENABLED = False
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_PART_LOSS_WEIGHT = 0.05
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LG_ENHANCE_ENABLED = False
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LG_ENHANCE_DIRECTION = 'global_to_local'  # global_to_local / local_to_global / bidirectional
_C.MODEL.OSNET_FUSION.STAGE3_STRIPE_LOCAL_LG_ENHANCE_INIT_SCALE = 0.05
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_ENABLED = False
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_NUM_STRIPES = 4
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_SHARE_PARAMS = True
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_MAMBA_DEPTH = 1
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_MAMBA_D_STATE = 8
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_MAMBA_D_CONV = 3
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_MAMBA_INIT_SCALE = 0.1
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_MAMBA_BIDIRECTIONAL = True
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_MLP_RATIO = 2.0
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_OVERLAP_RATIO = 0.0
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_CONTEXT_MIXER_TYPE = 'none'
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_CONTEXT_MIXER_KERNEL = 3
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_CONTEXT_MIXER_INIT_SCALE = 1e-3
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_RELIABILITY_GATE_ENABLED = False
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_RELIABILITY_HIDDEN_RATIO = 0.25
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_RELIABILITY_INIT_BIAS = 2.0
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_PART_DIM = 0
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_LOSS_WEIGHT = 0.2
_C.MODEL.OSNET_FUSION.STAGE4_STRIPE_LOCAL_INFER_WEIGHT = 0.3

# EMA Setting
_C.MODEL.EMA = CN()
_C.MODEL.EMA.ENABLED = False
_C.MODEL.EMA.DECAY = 0.9997

# SIE (Side Information Embedding) for MambaVision
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_XISHU = 1.5


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# 3Augment Data Augmentation
_C.INPUT.DA_AUGMENT = False
# Parallel Augmentation Mechanism (BA + CA + EA)
_C.INPUT.PAM = CN()
_C.INPUT.PAM.ENABLED = False
_C.INPUT.PAM.AUG_MODE = 'default'  # default: Fre-Reid PAM; pade: PADE-style BA/CA/EA transforms
_C.INPUT.PAM.CROP_PADDING = 30
_C.INPUT.PAM.CROP_SCALE = [0.08, 1.0]
_C.INPUT.PAM.CROP_RATIO = [0.75, 1.3333]
# Occlusion Simulation Based on Block Mixing. Training only.
_C.INPUT.OSBBM = CN()
_C.INPUT.OSBBM.ENABLED = False
_C.INPUT.OSBBM.PROB = 0.5
_C.INPUT.OSBBM.NUM_BLOCKS = 8
_C.INPUT.OSBBM.NUM_MIX_BLOCKS = 4
_C.INPUT.OSBBM.GRAY_PROB = 0.5
_C.INPUT.OSBBM.APPLY_TO = 'base'  # base, all
_C.INPUT.OSBBM.MIXED_LABEL = True
_C.INPUT.OSBBM.SCHEDULE = 'always'  # always, range, cycle
_C.INPUT.OSBBM.START_EPOCH = 1
_C.INPUT.OSBBM.END_EPOCH = 0  # 0 means SOLVER.MAX_EPOCHS
_C.INPUT.OSBBM.PERIOD_EPOCHS = 20
_C.INPUT.OSBBM.ON_EPOCHS = 10
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# Gradient Clipping threshold (0.0 means disabled)
_C.SOLVER.CLIP_GRAD_NORM = 1.0

# PAM loss: normalized L_BA + weight * (L_CA + L_EA)
_C.SOLVER.PAM_AUGMENTED_LOSS_WEIGHT = 1.0

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# Layer-wise learning rate decay (1.0 means no decay)
_C.SOLVER.LAYER_DECAY = 1.0

# Multiplier for Stage 4 (levels.3) learning rate (default 3.0)
_C.SOLVER.STAGE4_LR_FACTOR = 3.0

# Multiplier for SASF (state_fusion) learning rate (default 3.0)
_C.SOLVER.SASF_LR_FACTOR = 3.0

# SFM (SimpleFusionMamba) 损失权重 (HAT 风格λ)
_C.SOLVER.SFM_LAMBDA = 1.0        # 最终聚合级 (F34) 的损失权重
_C.SOLVER.SFM_LAMBDA_AUX = 0.2    # 中间聚合级 (F12, F23) 的损失权重

# Multiplier for SFM module learning rate (SFM is new, use 1.0 by default)
_C.SOLVER.SFM_LR_FACTOR = 1.0

# Multipliers for descriptor-level OSNet fusion experiments.
_C.SOLVER.OSNET_LR_FACTOR = 1.0
_C.SOLVER.OSNET_FUSION_LR_FACTOR = 2.0

# RATR (Ranking-aware Triplet Regularization) 损失配置
_C.SOLVER.RATR_ENABLED = False    # 是否启用 RATR
_C.SOLVER.RATR_LAMBDA = 1.0       # RATR 损失权重
_C.SOLVER.RATR_TAU = 0.1          # Kendall-tau 温度参数
_C.SOLVER.RATR_BRANCH_PAIR = 'mamba_osnet'  # mamba_osnet / mamba_fused / osnet_fused for OSNet fusion


# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Whether to concat all branch features at test time (for dual-branch mode)
_C.TEST.FEAT_CONCAT = False
# Feature mode for testing: 'fused' (default), 'main', 'fine', or 'concat'
_C.TEST.FEAT_MODE = 'fused'
# If False, do_inference evaluates only TEST.FEAT_MODE instead of every returned branch.
_C.TEST.EVAL_ALL_FEATS = True
# Extra eval-only weighted branch-normalized concat betas for OSNet fusion.
_C.TEST.BRANCH_NORM_BETAS = [0.25, 0.5, 0.63, 0.75]

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
