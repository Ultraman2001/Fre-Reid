import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class CICOBatchOcclusion(nn.Module):
    """
    Cross-Identity Consistent Occlusion for PAM.

    This follows COPE's CICO idea: samples with the same index modulo K share
    the same occlusion patch, so the occluder is decoupled from identity.
    """

    def __init__(
        self,
        num_patches,
        image_shape,
        group_size=4,
        min_area=1 / 5,
        max_area=1 / 2,
        min_aspect=0.3,
        max_aspect=None,
        occlusion_prob=1.0,
        device='cuda',
    ):
        super().__init__()
        self.num_patches = num_patches
        self.group_size = group_size
        self.image_h, self.image_w = image_shape
        self.min_area = min_area
        self.max_area = max_area
        self.occlusion_prob = occlusion_prob

        if max_aspect is None:
            max_aspect = 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        self.occlusion_info = []
        self.patch_list = nn.ParameterList()
        area = self.image_h * self.image_w
        device = torch.device(device)

        for _ in range(num_patches):
            while True:
                target_area = random.uniform(self.min_area, self.max_area) * area
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if 0 < w < self.image_w and 0 < h < self.image_h:
                    top = random.randint(0, self.image_h - h)
                    left = random.randint(0, self.image_w - w)
                    self.occlusion_info.append((top, left, h, w))
                    patch = nn.Parameter(torch.empty(3, h, w, device=device).normal_())
                    self.patch_list.append(patch)
                    break

        self.frozen()

    def forward(self, x):
        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f'CICO expects 3-channel images, got {c}')
        if (h, w) != (self.image_h, self.image_w):
            raise ValueError(f'CICO expects image size {(self.image_h, self.image_w)}, got {(h, w)}')

        out = x.clone()
        occ_mask = torch.zeros(b, h, w, device=x.device, dtype=x.dtype)

        for r in range(self.group_size):
            idxs = [i for i in range(b) if i % self.group_size == r]
            if not idxs:
                continue
            if random.random() > self.occlusion_prob:
                continue

            patch_idx = random.randint(0, self.num_patches - 1)
            top, left, patch_h, patch_w = self.occlusion_info[patch_idx]
            patch = self.patch_list[patch_idx].to(device=x.device, dtype=x.dtype)

            # COPE writes frozen random patches directly into normalized tensors.
            for i in idxs:
                out[i, :, top:top + patch_h, left:left + patch_w] = patch
                occ_mask[i, top:top + patch_h, left:left + patch_w] = 1

        return out, occ_mask

    def frozen(self):
        for patch in self.patch_list:
            patch.requires_grad = False


def downsample_occ_mask(occ_mask, feature_size):
    """Convert pixel-level occlusion masks to feature-map-level soft masks."""
    mask = occ_mask.unsqueeze(1).float()
    mask = F.interpolate(mask, size=feature_size, mode='area')
    return mask.squeeze(1)


def cico_occ_consistency_loss(feature_map, occ_mask, group_size=4, loss_type='mse'):
    """
    Align occlusion-region features among samples sharing the same CICO group.

    Args:
        feature_map: CICO feature map, shape (B, C, H, W).
        occ_mask: pixel-level or feature-level mask, shape (B, Hm, Wm).
    """
    b, c, h, w = feature_map.shape
    if occ_mask.shape[-2:] != (h, w):
        occ_mask = downsample_occ_mask(occ_mask, (h, w))

    mask = occ_mask.to(device=feature_map.device, dtype=feature_map.dtype).unsqueeze(1)
    denom = mask.flatten(2).sum(dim=2).clamp_min(1e-6)
    occ_feat = (feature_map * mask).flatten(2).sum(dim=2) / denom

    losses = []
    for r in range(group_size):
        idxs = [i for i in range(b) if i % group_size == r]
        if len(idxs) <= 1:
            continue
        group = occ_feat[idxs]
        valid = denom[idxs].squeeze(1) > 1e-6
        if valid.sum() <= 1:
            continue
        group = group[valid]
        if loss_type == 'cosine':
            group = F.normalize(group.float(), dim=1)
            sim = group @ group.t()
            pair_mask = ~torch.eye(group.size(0), dtype=torch.bool, device=group.device)
            losses.append((1.0 - sim[pair_mask]).mean())
        else:
            diff = group.unsqueeze(1).float() - group.unsqueeze(0).float()
            dist = (diff ** 2).mean(dim=-1)
            pair_mask = ~torch.eye(group.size(0), dtype=torch.bool, device=group.device)
            losses.append(dist[pair_mask].mean())

    if not losses:
        return feature_map.new_tensor(0.0)
    return torch.stack(losses).mean()
