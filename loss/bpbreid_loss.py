import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import TripletLoss


GLOBAL = "globl"
FOREGROUND = "foreg"
CONCAT_PARTS = "conct"
PARTS = "parts"


def cross_entropy_label_smooth(inputs, targets, epsilon=0.1, weights=None):
    log_probs = F.log_softmax(inputs, dim=1)
    num_classes = inputs.size(1)
    targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets_one_hot = (1 - epsilon) * targets_one_hot + epsilon / num_classes
    loss = (-targets_one_hot * log_probs).sum(dim=1)
    if weights is not None:
        weights = weights.float()
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1e-6)
    return loss.mean()


def pairwise_part_distance(embeddings):
    # embeddings: [N, M, D] -> [M, N, N]
    x = embeddings.float().permute(1, 0, 2)
    squared = (x ** 2).sum(dim=2, keepdim=True)
    dist = squared + squared.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    dist = dist.clamp_min(0.0)
    zero_mask = dist.eq(0.0).float()
    return (dist + zero_mask * 1e-16).sqrt() * (1.0 - zero_mask)


def masked_mean(dist, visibility):
    if visibility is None:
        return dist.mean(dim=0), torch.ones_like(dist[0], dtype=torch.bool)

    vis = visibility.float().t()
    weights = torch.sqrt(vis.unsqueeze(2) * vis.unsqueeze(1))
    denom = weights.sum(dim=0)
    pairwise = (dist * weights).sum(dim=0) / denom.clamp_min(1e-6)
    valid = denom > 0
    return pairwise, valid


class PartAveragedTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        if margin is None or margin <= 0:
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels, parts_visibility=None):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)

        dist_parts = pairwise_part_distance(embeddings)
        dist, valid_pair_mask = masked_mean(dist_parts, parts_visibility)
        n = dist.size(0)
        labels_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        eye = torch.eye(n, dtype=torch.bool, device=labels.device)
        pos_mask = labels_equal & ~eye & valid_pair_mask
        neg_mask = ~labels_equal & valid_pair_mask

        max_value = dist.max().detach() + 1.0
        hardest_pos = (dist * pos_mask.float() - (~pos_mask).float()).max(dim=1)[0]
        hardest_neg = (dist * neg_mask.float() + (~neg_mask).float() * max_value).min(dim=1)[0]
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            zero = embeddings.sum() * 0.0
            return zero, torch.tensor(0.0, device=labels.device), torch.tensor(0.0, device=labels.device)

        dist_ap = hardest_pos[valid]
        dist_an = hardest_neg[valid]
        y = torch.ones_like(dist_an)
        if self.margin is None or self.margin <= 0:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)

        trivial_margin = float(self.margin) if self.margin is not None and self.margin > 0 else 0.3
        trivial = F.relu(dist_ap - dist_an + trivial_margin).eq(0.0).float().mean().detach()
        valid_ratio = valid.float().mean().detach()
        return loss, trivial, valid_ratio


class BodyPartAttentionLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pixels_cls_scores, target_masks):
        target_masks = F.interpolate(
            target_masks.float(),
            size=pixels_cls_scores.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        targets = target_masks.argmax(dim=1)
        loss = F.cross_entropy(
            pixels_cls_scores.float(),
            targets,
            label_smoothing=self.label_smoothing,
        )
        pred = pixels_cls_scores.argmax(dim=1)
        acc = pred.eq(targets).float().mean()
        return loss, acc


class BPBreIDLoss(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.num_classes = num_classes
        bp_cfg = cfg.MODEL.BPBREID
        self.id_epsilon = 0.1 if cfg.MODEL.IF_LABELSMOOTH == "on" else 0.0
        self.use_visibility_scores = bool(
            getattr(bp_cfg, "MASK_FILTERING_TRAINING", getattr(bp_cfg, "USE_VISIBILITY_SCORES", True))
        )
        self.loss_weights = {
            GLOBAL: {
                "id": float(bp_cfg.GLOBAL_ID_WEIGHT),
                "tr": float(bp_cfg.GLOBAL_TRIPLET_WEIGHT),
            },
            FOREGROUND: {
                "id": float(bp_cfg.FOREGROUND_ID_WEIGHT),
                "tr": float(bp_cfg.FOREGROUND_TRIPLET_WEIGHT),
            },
            CONCAT_PARTS: {
                "id": float(bp_cfg.CONCAT_PARTS_ID_WEIGHT),
                "tr": float(bp_cfg.CONCAT_PARTS_TRIPLET_WEIGHT),
            },
            PARTS: {
                "id": float(bp_cfg.PARTS_ID_WEIGHT),
                "tr": float(bp_cfg.PARTS_TRIPLET_WEIGHT),
            },
        }
        self.part_triplet = PartAveragedTripletLoss(float(bp_cfg.PART_TRIPLET_MARGIN))
        self.attention_loss = BodyPartAttentionLoss(label_smoothing=float(bp_cfg.ATTENTION_LABEL_SMOOTHING))
        self.attention_loss_weight = float(bp_cfg.ATTENTION_LOSS_WEIGHT)
        self.anchor_enabled = bool(getattr(bp_cfg, "ANCHOR_ENABLED", False))
        self.anchor_id_weight = float(getattr(bp_cfg, "ANCHOR_ID_WEIGHT", 1.0))
        self.anchor_triplet_weight = float(getattr(bp_cfg, "ANCHOR_TRIPLET_WEIGHT", 1.0))
        anchor_margin = None if cfg.MODEL.NO_MARGIN else float(cfg.SOLVER.MARGIN)
        self.anchor_triplet = TripletLoss(anchor_margin)

    def _id_loss(self, cls_scores, visibility_scores, targets):
        weights = None
        if cls_scores.dim() == 3:
            n, m, _ = cls_scores.shape
            cls_scores = cls_scores.flatten(0, 1)
            targets = targets.unsqueeze(1).expand(n, m).flatten(0, 1)
            visibility_scores = visibility_scores.flatten(0, 1)

        if self.use_visibility_scores and visibility_scores.dtype is torch.bool:
            if not visibility_scores.any():
                return cls_scores.sum() * 0.0, torch.tensor(0.0, device=cls_scores.device)
            cls_scores = cls_scores[visibility_scores]
            targets = targets[visibility_scores]
        elif self.use_visibility_scores:
            weights = visibility_scores.flatten().float()

        loss = cross_entropy_label_smooth(cls_scores.float(), targets, self.id_epsilon, weights)
        acc = cls_scores.argmax(dim=1).eq(targets).float().mean().detach()
        return loss, acc

    def _triplet_loss(self, embeddings, visibility_scores, targets):
        visibility = visibility_scores if self.use_visibility_scores else None
        return self.part_triplet(embeddings.float(), targets, visibility)

    def forward(self, output, targets, target_masks=None):
        embeddings = output["embeddings"]
        visibility = output["visibility_scores"]
        cls_scores = output["id_cls_scores"]

        losses = []
        detail = {"bp_loss": 1.0}
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            id_weight = self.loss_weights[key]["id"]
            if id_weight > 0:
                loss_id, acc = self._id_loss(cls_scores[key], visibility[key], targets)
                losses.append(id_weight * loss_id)
                detail["{}_id".format(key)] = loss_id.item()
                detail["{}_acc".format(key)] = acc.item()

            tr_weight = self.loss_weights[key]["tr"]
            if tr_weight > 0:
                loss_tri, trivial_ratio, valid_ratio = self._triplet_loss(embeddings[key], visibility[key], targets)
                losses.append(tr_weight * loss_tri)
                detail["{}_tri".format(key)] = loss_tri.item()
                detail["{}_tri_valid".format(key)] = valid_ratio.item()
                detail["{}_tri_trivial".format(key)] = trivial_ratio.item()

        if self.attention_loss_weight > 0 and target_masks is not None:
            att_loss, att_acc = self.attention_loss(output["pixels_cls_scores"], target_masks)
            losses.append(self.attention_loss_weight * att_loss)
            detail["pixels_ce"] = att_loss.item()
            detail["pixels_acc"] = att_acc.item()

        if self.anchor_enabled:
            if "anchor_embedding" not in output or "anchor_cls_score" not in output:
                raise KeyError("BPBreID GeM anchor is enabled but its training outputs are missing")
            if self.anchor_id_weight > 0:
                anchor_id = cross_entropy_label_smooth(
                    output["anchor_cls_score"].float(),
                    targets,
                    self.id_epsilon,
                )
                losses.append(self.anchor_id_weight * anchor_id)
                detail["anchor_id"] = anchor_id.item()
            if self.anchor_triplet_weight > 0:
                anchor_tri = self.anchor_triplet(output["anchor_embedding"].float(), targets)[0]
                losses.append(self.anchor_triplet_weight * anchor_tri)
                detail["anchor_tri"] = anchor_tri.item()

        if not losses:
            total = output["concat"].sum() * 0.0
        else:
            total = torch.stack(losses).sum()
        return total, detail
