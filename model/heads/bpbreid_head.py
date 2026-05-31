import torch
import torch.nn as nn
import torch.nn.functional as F


GLOBAL = "globl"
BACKGROUND = "backg"
FOREGROUND = "foreg"
CONCAT_PARTS = "conct"
PARTS = "parts"
BN_GLOBAL = "bn_globl"
BN_BACKGROUND = "bn_backg"
BN_FOREGROUND = "bn_foreg"
BN_CONCAT_PARTS = "bn_conct"
BN_PARTS = "bn_parts"


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class BNClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        self.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.bn(x)
        cls_score = self.classifier(feat)
        return feat, cls_score


class PixelToPartClassifier(nn.Module):
    def __init__(self, in_dim, parts_num):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_dim)
        self.classifier = nn.Conv2d(in_dim, parts_num + 1, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x):
        return self.classifier(self.bn(x))


class AfterPoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout_p and dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))
        self.layers = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if x.dim() == 3:
            n, m, d = x.shape
            x = x.flatten(0, 1)
            x = self.layers(x)
            return x.view(n, m, -1)
        return self.layers(x)


class HighResolutionFeatureAdapter(nn.Module):
    def __init__(self, shallow_dim, semantic_dim, output_dim):
        super().__init__()
        self.shallow_proj = nn.Sequential(
            nn.Conv2d(shallow_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                output_dim,
                output_dim,
                kernel_size=3,
                padding=1,
                groups=output_dim,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.apply(weights_init_kaiming)

    def forward(self, semantic_features, shallow_features):
        semantic_features = self.semantic_proj(semantic_features)
        semantic_features = F.interpolate(
            semantic_features,
            size=shallow_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(self.shallow_proj(shallow_features) + semantic_features)


def global_masked_average_pooling(features, part_masks):
    if part_masks.dim() == 3:
        part_masks = part_masks.unsqueeze(1)
    weighted = features.float().unsqueeze(1) * part_masks.float().unsqueeze(2)
    return weighted.mean(dim=(-1, -2)).to(features.dtype)


def global_weighted_average_pooling(features, part_masks):
    if part_masks.dim() == 3:
        part_masks = part_masks.unsqueeze(1)
    features_f = features.float()
    masks_f = part_masks.float()
    weighted = features_f.unsqueeze(1) * masks_f.unsqueeze(2)
    pooled = weighted.sum(dim=(-1, -2))
    denom = masks_f.sum(dim=(-1, -2)).clamp_min(1e-6).unsqueeze(-1)
    return (pooled / denom).to(features.dtype)


class BPBreIDHead(nn.Module):
    def __init__(self, in_dim, num_classes, cfg, shallow_dim=None, global_pooling=None):
        super().__init__()
        bp_cfg = cfg.MODEL.BPBREID
        self.parts_num = int(bp_cfg.PARTS_NUM)
        self.dim_reduce_output = int(bp_cfg.DIM_REDUCE)
        self.global_pooling = global_pooling if global_pooling is not None else nn.AdaptiveAvgPool2d(1)
        self.shared_parts_id_classifier = bool(bp_cfg.SHARED_PARTS_ID_CLASSIFIER)
        self.training_binary_visibility_score = bool(bp_cfg.TRAINING_BINARY_VISIBILITY)
        self.testing_binary_visibility_score = bool(bp_cfg.TESTING_BINARY_VISIBILITY)
        self.mask_filtering_testing = bool(
            getattr(bp_cfg, "MASK_FILTERING_TESTING", getattr(bp_cfg, "USE_VISIBILITY_SCORES", True))
        )
        self.test_use_target_segmentation = bp_cfg.TEST_USE_TARGET_SEGMENTATION
        self.test_embeddings = list(bp_cfg.TEST_EMBEDDINGS)

        self.high_resolution_enabled = bool(getattr(bp_cfg, "HIGH_RESOLUTION_ENABLED", False))
        self.pooling_feature_source = str(
            getattr(bp_cfg, "POOLING_FEATURE_SOURCE", "semantic")
        ).lower()
        if self.pooling_feature_source not in ("semantic", "high_resolution"):
            raise ValueError(
                "BPBreID POOLING_FEATURE_SOURCE must be 'semantic' or 'high_resolution'"
            )

        attention_dim = in_dim
        pooling_dim = in_dim
        if self.high_resolution_enabled:
            if shallow_dim is None:
                raise ValueError("BPBreID high-resolution adapter requires a shallow feature dimension")
            attention_dim = int(bp_cfg.HIGH_RESOLUTION_DIM)
            self.high_resolution_adapter = HighResolutionFeatureAdapter(
                shallow_dim,
                in_dim,
                attention_dim,
            )
            if self.pooling_feature_source == "high_resolution":
                pooling_dim = attention_dim
        else:
            if self.pooling_feature_source == "high_resolution":
                raise ValueError(
                    "BPBreID POOLING_FEATURE_SOURCE='high_resolution' requires "
                    "HIGH_RESOLUTION_ENABLED=True"
                )
            self.high_resolution_adapter = None

        self.pixel_classifier = PixelToPartClassifier(attention_dim, self.parts_num)

        if self.dim_reduce_output > 0:
            dropout = float(bp_cfg.DIM_REDUCE_DROPOUT)
            self.global_reduce = AfterPoolingDimReduceLayer(pooling_dim, self.dim_reduce_output, dropout)
            self.background_reduce = AfterPoolingDimReduceLayer(pooling_dim, self.dim_reduce_output, dropout)
            self.foreground_reduce = AfterPoolingDimReduceLayer(pooling_dim, self.dim_reduce_output, dropout)
            self.parts_reduce = AfterPoolingDimReduceLayer(pooling_dim, self.dim_reduce_output, dropout)
            out_dim = self.dim_reduce_output
        else:
            self.global_reduce = nn.Identity()
            self.background_reduce = nn.Identity()
            self.foreground_reduce = nn.Identity()
            self.parts_reduce = nn.Identity()
            out_dim = pooling_dim
            self.dim_reduce_output = pooling_dim

        self.global_identity_classifier = BNClassifier(out_dim, num_classes)
        self.background_identity_classifier = BNClassifier(out_dim, num_classes)
        self.foreground_identity_classifier = BNClassifier(out_dim, num_classes)
        self.concat_parts_identity_classifier = BNClassifier(self.parts_num * out_dim, num_classes)

        if self.shared_parts_id_classifier:
            self.parts_identity_classifier = BNClassifier(out_dim, num_classes)
        else:
            self.parts_identity_classifier = nn.ModuleList(
                [BNClassifier(out_dim, num_classes) for _ in range(self.parts_num)]
            )

    def _parts_identity_classification(self, parts_embeddings):
        n, k, d = parts_embeddings.shape
        if self.shared_parts_id_classifier:
            flat = parts_embeddings.flatten(0, 1)
            bn_part_embeddings, part_cls_score = self.parts_identity_classifier(flat)
            return bn_part_embeddings.view(n, k, d), part_cls_score.view(n, k, -1)

        embeddings = []
        cls_scores = []
        for i, classifier in enumerate(self.parts_identity_classifier):
            bn_part_embeddings, part_cls_score = classifier(parts_embeddings[:, i])
            embeddings.append(bn_part_embeddings.unsqueeze(1))
            cls_scores.append(part_cls_score.unsqueeze(1))
        return torch.cat(embeddings, dim=1), torch.cat(cls_scores, dim=1)

    def _apply_target_segmentation(self, parts_masks, background_masks, external_parts_masks, size):
        mode = self.test_use_target_segmentation
        if self.training or external_parts_masks is None or mode == "none":
            return parts_masks, background_masks

        external = F.interpolate(
            external_parts_masks.float(),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        if mode == "hard":
            target_segmentation_mask = external[:, 1:].amax(dim=1) > external[:, 0]
            background_masks = ~target_segmentation_mask
            parts_masks = parts_masks.masked_fill(background_masks.unsqueeze(1), 1e-12)
        elif mode == "soft":
            parts_masks = parts_masks * external[:, 1:]
        else:
            raise ValueError("Unsupported TEST_USE_TARGET_SEGMENTATION: {}".format(mode))
        return parts_masks, background_masks

    def _visibility_scores(self, part_probabilities):
        binary = self.training_binary_visibility_score if self.training else self.testing_binary_visibility_score
        if binary:
            predictions = part_probabilities.argmax(dim=1)
            one_hot = F.one_hot(predictions, self.parts_num + 1).permute(0, 3, 1, 2)
            visibility = one_hot.amax(dim=(2, 3)).to(torch.bool)
        else:
            visibility = part_probabilities.amax(dim=(2, 3))

        background_visibility = visibility[:, 0]
        foreground_visibility = visibility.amax(dim=1)
        parts_visibility = visibility[:, 1:]
        concat_parts_visibility = foreground_visibility
        global_visibility = torch.ones_like(foreground_visibility)
        return {
            GLOBAL: global_visibility,
            BACKGROUND: background_visibility,
            FOREGROUND: foreground_visibility,
            CONCAT_PARTS: concat_parts_visibility,
            PARTS: parts_visibility,
            BN_GLOBAL: global_visibility,
            BN_BACKGROUND: background_visibility,
            BN_FOREGROUND: foreground_visibility,
            BN_CONCAT_PARTS: concat_parts_visibility,
            BN_PARTS: parts_visibility,
        }

    def _select_test_features(self, embeddings, visibility_scores):
        features = []
        visibility = []
        for key in self.test_embeddings:
            if key not in embeddings:
                raise KeyError("Unknown BPBreID test embedding '{}'".format(key))
            feat = embeddings[key]
            vis_key = key
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)
            if key.startswith("bn_"):
                vis_key = key[3:]
            vis = visibility_scores[vis_key]
            if vis.dim() == 1:
                vis = vis.unsqueeze(1)
            features.append(feat)
            visibility.append(vis)
        features = torch.cat(features, dim=1)
        visibility = torch.cat(visibility, dim=1)
        if not self.mask_filtering_testing:
            visibility = torch.ones_like(visibility, dtype=torch.bool)
        return features, visibility

    def forward(self, spatial_features, external_parts_masks=None, shallow_features=None):
        pooling_features = spatial_features
        attention_features = spatial_features
        if self.high_resolution_adapter is not None:
            if shallow_features is None:
                raise ValueError("BPBreID high-resolution adapter requires shallow features")
            attention_features = self.high_resolution_adapter(spatial_features, shallow_features)
            if self.pooling_feature_source == "high_resolution":
                pooling_features = attention_features

        _, _, hf, wf = attention_features.shape
        pixels_cls_scores = self.pixel_classifier(attention_features)
        pixels_parts_probabilities = F.softmax(pixels_cls_scores.float(), dim=1).to(attention_features.dtype)

        background_masks = pixels_parts_probabilities[:, 0]
        parts_masks = pixels_parts_probabilities[:, 1:]
        parts_masks, background_masks = self._apply_target_segmentation(
            parts_masks,
            background_masks,
            external_parts_masks,
            (hf, wf),
        )

        if parts_masks.shape[-2:] != pooling_features.shape[-2:]:
            pooling_size = pooling_features.shape[-2:]
            parts_masks = F.interpolate(
                parts_masks.float(),
                size=pooling_size,
                mode="bilinear",
                align_corners=True,
            ).to(pooling_features.dtype)
            background_masks = F.interpolate(
                background_masks.unsqueeze(1).float(),
                size=pooling_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(1).to(pooling_features.dtype)

        foreground_masks = parts_masks.amax(dim=1)
        global_masks = torch.ones_like(foreground_masks)
        visibility_scores = self._visibility_scores(pixels_parts_probabilities)

        global_embeddings = self.global_pooling(pooling_features).flatten(1)
        background_embeddings = global_masked_average_pooling(
            pooling_features,
            background_masks.unsqueeze(1),
        ).flatten(1)
        foreground_embeddings = global_masked_average_pooling(
            pooling_features,
            foreground_masks.unsqueeze(1),
        ).flatten(1)
        parts_embeddings = global_weighted_average_pooling(pooling_features, parts_masks)

        global_embeddings = self.global_reduce(global_embeddings)
        background_embeddings = self.background_reduce(background_embeddings)
        foreground_embeddings = self.foreground_reduce(foreground_embeddings)
        parts_embeddings = self.parts_reduce(parts_embeddings)
        concat_parts_embeddings = parts_embeddings.flatten(1, 2)

        bn_global_embeddings, global_cls_score = self.global_identity_classifier(global_embeddings)
        bn_background_embeddings, background_cls_score = self.background_identity_classifier(background_embeddings)
        bn_foreground_embeddings, foreground_cls_score = self.foreground_identity_classifier(foreground_embeddings)
        bn_concat_parts_embeddings, concat_parts_cls_score = self.concat_parts_identity_classifier(concat_parts_embeddings)
        bn_parts_embeddings, parts_cls_score = self._parts_identity_classification(parts_embeddings)

        embeddings = {
            GLOBAL: global_embeddings,
            BACKGROUND: background_embeddings,
            FOREGROUND: foreground_embeddings,
            CONCAT_PARTS: concat_parts_embeddings,
            PARTS: parts_embeddings,
            BN_GLOBAL: bn_global_embeddings,
            BN_BACKGROUND: bn_background_embeddings,
            BN_FOREGROUND: bn_foreground_embeddings,
            BN_CONCAT_PARTS: bn_concat_parts_embeddings,
            BN_PARTS: bn_parts_embeddings,
        }
        id_cls_scores = {
            GLOBAL: global_cls_score,
            BACKGROUND: background_cls_score,
            FOREGROUND: foreground_cls_score,
            CONCAT_PARTS: concat_parts_cls_score,
            PARTS: parts_cls_score,
        }
        masks = {
            GLOBAL: global_masks,
            BACKGROUND: background_masks,
            FOREGROUND: foreground_masks,
            CONCAT_PARTS: foreground_masks,
            PARTS: parts_masks,
        }
        test_features, test_visibility = self._select_test_features(embeddings, visibility_scores)

        return {
            "type": "bpbreid",
            "embeddings": embeddings,
            "visibility_scores": visibility_scores,
            "id_cls_scores": id_cls_scores,
            "pixels_cls_scores": pixels_cls_scores,
            "spatial_features": pooling_features,
            "attention_features": attention_features,
            "masks": masks,
            "bp_features": test_features,
            "bp_visibility": test_visibility,
            "concat": test_features.flatten(1, 2),
        }
