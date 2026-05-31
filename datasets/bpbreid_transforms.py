import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


PIFPAF_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

PIFPAF_JOINTS = [
    "left_ankle_to_left_knee", "left_knee_to_left_hip",
    "right_ankle_to_right_knee", "right_knee_to_right_hip",
    "left_hip_to_right_hip", "left_shoulder_to_left_hip",
    "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder",
    "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
    "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
    "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye",
    "left_eye_to_left_ear", "right_eye_to_right_ear",
    "left_ear_to_left_shoulder", "right_ear_to_right_shoulder",
]

PIFPAF_PARTS = PIFPAF_KEYPOINTS + PIFPAF_JOINTS
PIFPAF_PARTS_MAP = {name: idx for idx, name in enumerate(PIFPAF_PARTS)}

PIFPAF_GROUPINGS = {
    "four": [
        ("head_mask", [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye",
            "left_eye_to_left_ear", "right_eye_to_right_ear",
            "left_ear_to_left_shoulder", "right_ear_to_right_shoulder",
        ]),
        ("torso_mask", [
            "left_hip", "right_hip", "left_hip_to_right_hip",
            "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
            "left_shoulder_to_right_shoulder",
        ]),
        ("arms_mask", [
            "left_shoulder", "left_elbow", "left_wrist",
            "left_shoulder_to_left_elbow", "left_elbow_to_left_wrist",
            "right_shoulder", "right_elbow", "right_wrist",
            "right_shoulder_to_right_elbow", "right_elbow_to_right_wrist",
        ]),
        ("legs_mask", [
            "left_knee", "left_ankle", "left_ankle_to_left_knee",
            "left_knee_to_left_hip", "right_knee", "right_ankle",
            "right_ankle_to_right_knee", "right_knee_to_right_hip",
        ]),
    ],
    "five_v": [
        ("head_mask", [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye",
            "left_eye_to_left_ear", "right_eye_to_right_ear",
            "left_ear_to_left_shoulder", "right_ear_to_right_shoulder",
        ]),
        ("upper_arms_torso_mask", [
            "left_elbow", "right_elbow",
            "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
            "left_shoulder", "right_shoulder", "left_shoulder_to_right_shoulder",
        ]),
        ("lower_arms_torso_mask", [
            "left_wrist", "right_wrist",
            "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
            "left_hip", "right_hip", "right_shoulder_to_right_hip",
        ]),
        ("legs_mask", [
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle_to_left_knee", "left_knee_to_left_hip",
            "right_ankle_to_right_knee", "right_knee_to_right_hip",
        ]),
        ("feet_mask", ["left_ankle", "right_ankle"]),
    ],
    "six": [
        ("head_mask", [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye",
            "left_eye_to_left_ear", "right_eye_to_right_ear",
            "left_ear_to_left_shoulder", "right_ear_to_right_shoulder",
        ]),
        ("left_arm_mask", [
            "left_shoulder", "left_elbow", "left_wrist",
            "left_shoulder_to_left_elbow", "left_elbow_to_left_wrist",
        ]),
        ("right_arm_mask", [
            "right_shoulder", "right_elbow", "right_wrist",
            "right_shoulder_to_right_elbow", "right_elbow_to_right_wrist",
        ]),
        ("torso_mask", [
            "left_hip", "right_hip", "left_hip_to_right_hip",
            "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
            "left_shoulder_to_right_shoulder",
        ]),
        ("left_leg_mask", [
            "left_knee", "left_ankle", "left_ankle_to_left_knee",
            "left_knee_to_left_hip", "left_hip_to_right_hip",
        ]),
        ("right_leg_mask", [
            "right_knee", "right_ankle", "right_ankle_to_right_knee",
            "right_knee_to_right_hip",
        ]),
    ],
    "eight": [
        ("head_mask", [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye",
            "left_eye_to_left_ear", "right_eye_to_right_ear",
            "left_ear_to_left_shoulder", "right_ear_to_right_shoulder",
        ]),
        ("left_arm_mask", [
            "left_shoulder", "left_elbow", "left_wrist",
            "left_shoulder_to_left_elbow", "left_elbow_to_left_wrist",
        ]),
        ("right_arm_mask", [
            "right_shoulder", "right_elbow", "right_wrist",
            "right_shoulder_to_right_elbow", "right_elbow_to_right_wrist",
        ]),
        ("torso_mask", [
            "left_hip", "right_hip", "left_hip_to_right_hip",
            "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
            "left_shoulder_to_right_shoulder",
        ]),
        ("left_leg_mask", [
            "left_knee", "left_ankle_to_left_knee",
            "left_knee_to_left_hip", "left_hip_to_right_hip",
        ]),
        ("right_leg_mask", [
            "right_knee", "right_ankle_to_right_knee",
            "right_knee_to_right_hip",
        ]),
        ("left_feet_mask", ["left_ankle"]),
        ("right_feet_mask", ["right_ankle"]),
    ],
}


def infer_masks_path(img_path, masks_dir, masks_base_dir, masks_suffix, masks_root=""):
    if masks_root:
        root = masks_root
    else:
        root = osp.dirname(osp.dirname(img_path))
    split_name = osp.basename(osp.dirname(img_path))
    stem = osp.splitext(osp.basename(img_path))[0]
    return osp.join(root, masks_base_dir, masks_dir, split_name, stem + masks_suffix)


def read_parsing_mask(mask_path, expected_channels):
    if not osp.exists(mask_path):
        raise IOError('Masks file "{}" does not exist'.format(mask_path))

    masks = np.load(mask_path)
    if masks.ndim != 3:
        raise ValueError(
            'Expected a 3-D PifPaf mask stack at "{}", got shape {}'.format(mask_path, masks.shape)
        )

    channel_axes = [axis for axis, size in enumerate(masks.shape) if size == expected_channels]
    if len(channel_axes) != 1:
        raise ValueError(
            'Expected {} raw PifPaf channels at "{}", got shape {}. '
            'The official BPBreID Duke pifpaf_maskrcnn_filtering labels contain 36 foreground '
            'channels without a background channel. Check that the correct label package was '
            'extracted under the configured masks directory.'.format(
                expected_channels,
                mask_path,
                masks.shape,
            )
        )

    masks = np.moveaxis(masks, channel_axes[0], 0)
    masks = torch.from_numpy(np.ascontiguousarray(masks)).float()
    return masks.clamp_(0.0, 1.0)


def group_pifpaf_masks(masks, preprocess):
    preprocess = preprocess.lower()
    if preprocess in ("none", "identity"):
        return masks
    if preprocess not in PIFPAF_GROUPINGS:
        raise ValueError("Unsupported PifPaf mask preprocess: {}".format(preprocess))
    if masks.size(0) != len(PIFPAF_PARTS):
        raise ValueError(
            "PifPaf mask preprocessing '{}' requires {} raw channels, got {}".format(
                preprocess,
                len(PIFPAF_PARTS),
                masks.size(0),
            )
        )

    grouped = []
    for _, part_names in PIFPAF_GROUPINGS[preprocess]:
        indices = [PIFPAF_PARTS_MAP[name] for name in part_names]
        grouped.append(masks[indices].amax(dim=0).clamp(0.0, 1.0))
    return torch.stack(grouped, dim=0)


def add_background_mask(masks, strategy, softmax_weight, threshold):
    if strategy == "sum":
        background = (1.0 - masks.sum(dim=0)).clamp(0.0, 1.0)
    elif strategy == "threshold":
        background = (masks.amax(dim=0) < threshold).float()
    elif strategy == "diff_from_max":
        background = (1.0 - masks.amax(dim=0)).clamp(0.0, 1.0)
    else:
        raise ValueError("Unsupported background strategy: {}".format(strategy))

    masks = torch.cat([background.unsqueeze(0), masks], dim=0)
    if softmax_weight > 0:
        masks = F.softmax(masks * softmax_weight, dim=0)
    else:
        masks = masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)
    return masks


class BPReIDTransform:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.size = tuple(cfg.INPUT.SIZE_TRAIN if is_train else cfg.INPUT.SIZE_TEST)
        self.flip_prob = cfg.INPUT.PROB if is_train else 0.0
        self.padding = cfg.INPUT.PADDING if is_train else 0
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD
        self.random_erasing_prob = cfg.INPUT.RE_PROB if is_train else 0.0

        bp_cfg = cfg.MODEL.BPBREID
        self.masks_dir = bp_cfg.MASKS_DIR
        self.masks_base_dir = bp_cfg.MASKS_BASE_DIR
        self.masks_suffix = bp_cfg.MASKS_SUFFIX
        self.masks_root = bp_cfg.MASKS_ROOT
        self.masks_source_channels = int(bp_cfg.MASKS_SOURCE_CHANNELS)
        self.masks_preprocess = bp_cfg.MASKS_PREPROCESS
        self.softmax_weight = bp_cfg.MASKS_SOFTMAX_WEIGHT
        self.background_strategy = bp_cfg.MASKS_BACKGROUND_STRATEGY
        self.mask_filtering_threshold = bp_cfg.MASKS_FILTERING_THRESHOLD
        self.mask_scale = max(1, int(bp_cfg.MASK_SCALE))
        if self.parts_num != int(bp_cfg.PARTS_NUM):
            raise ValueError(
                "BPBreID PARTS_NUM={} does not match MASKS_PREPROCESS='{}' ({} parts)".format(
                    bp_cfg.PARTS_NUM,
                    self.masks_preprocess,
                    self.parts_num,
                )
            )

    def infer_masks_path(self, img_path):
        return infer_masks_path(
            img_path,
            self.masks_dir,
            self.masks_base_dir,
            self.masks_suffix,
            self.masks_root,
        )

    def validate_sample(self, img_path):
        mask_path = self.infer_masks_path(img_path)
        source_mask = read_parsing_mask(mask_path, self.masks_source_channels)
        processed_mask = self._process_mask(source_mask)
        expected_channels = self.parts_num + 1
        if processed_mask.size(0) != expected_channels:
            raise ValueError(
                "BPBreID mask preprocessing produced {} channels for '{}', expected {} "
                "(background + {} parts)".format(
                    processed_mask.size(0),
                    mask_path,
                    expected_channels,
                    self.parts_num,
                )
            )
        return mask_path, tuple(source_mask.shape), tuple(processed_mask.shape)

    @staticmethod
    def _resize_mask(mask, size, mode="bilinear"):
        mask = mask.unsqueeze(0)
        if mode == "nearest":
            mask = F.interpolate(mask, size=size, mode=mode)
        else:
            mask = F.interpolate(mask, size=size, mode=mode, align_corners=False)
        return mask.squeeze(0)

    def _crop(self, image, mask):
        target_h, target_w = self.size
        width, height = image.size
        if height == target_h and width == target_w:
            return image, mask
        top = random.randint(0, height - target_h)
        left = random.randint(0, width - target_w)
        image = TF.crop(image, top, left, target_h, target_w)
        mask = mask[:, top:top + target_h, left:left + target_w]
        return image, mask

    def _process_mask(self, mask):
        mask = group_pifpaf_masks(mask, self.masks_preprocess)
        mask = add_background_mask(
            mask,
            self.background_strategy,
            self.softmax_weight,
            self.mask_filtering_threshold,
        )
        h = max(1, self.size[0] // self.mask_scale)
        w = max(1, self.size[1] // self.mask_scale)
        return self._resize_mask(mask, (h, w), mode="nearest")

    def _random_erase(self, image, mask):
        if random.random() >= self.random_erasing_prob:
            return image, mask

        height, width = image.shape[-2:]
        erase_h = random.randint(max(1, int(height * 0.15)), max(1, int(height * 0.65)))
        erase_w = random.randint(max(1, int(width * 0.15)), max(1, int(width * 0.65)))
        top = random.randint(0, height - erase_h)
        left = random.randint(0, width - erase_w)

        fill = image.new_tensor(self.mean).view(-1, 1, 1)
        image[:, top:top + erase_h, left:left + erase_w] = fill
        mask[:, top:top + erase_h, left:left + erase_w] = 0
        return image, mask

    def __call__(self, image, img_path):
        mask = read_parsing_mask(
            self.infer_masks_path(img_path),
            self.masks_source_channels,
        )

        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        mask = self._resize_mask(mask, self.size, mode="bilinear")

        if self.is_train and random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = torch.flip(mask, dims=[2])

        if self.is_train and self.padding > 0:
            image = TF.pad(image, self.padding, fill=0)
            mask = F.pad(mask, (self.padding, self.padding, self.padding, self.padding))
            image, mask = self._crop(image, mask)

        image = TF.to_tensor(image)
        image, mask = self._random_erase(image, mask)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = self._process_mask(mask)
        return image, mask

    @property
    def parts_num(self):
        preprocess = self.masks_preprocess.lower()
        if preprocess in PIFPAF_GROUPINGS:
            return len(PIFPAF_GROUPINGS[preprocess])
        return 36
