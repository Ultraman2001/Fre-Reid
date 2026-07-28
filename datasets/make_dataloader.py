import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from .bases import DualViewImageDataset, ImageDataset, ParallelAugmentationImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def pam_train_collate_fn(batch):
    imgs_base, imgs_crop, imgs_erase, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (
        torch.stack(imgs_base, dim=0),
        torch.stack(imgs_crop, dim=0),
        torch.stack(imgs_erase, dim=0),
        pids,
        camids,
        viewids,
    )


class DualViewTrainCollator:
    """Route clean/erased/cropped candidates to heterogeneous backbones.

    The configuration namespace and returned two-view tuple are intentionally
    separate from the legacy PAM BA/CA/EA implementation.
    """

    _MODES = {
        'shared',
        'diffmask',
        'independent',
        'anticorr',
        'fixed',
        'anchor',
        'state_sample',
    }
    _DIRECTIONS = {'mamba_erased', 'osnet_erased'}
    _APPEARANCE_TYPES = {'none', 'color', 'grayscale', 'blur'}
    _APPEARANCE_TARGETS = {'mamba', 'osnet', 'shared', 'random_one'}

    def __init__(self, cfg):
        dual_cfg = cfg.INPUT.DUAL_VIEW
        self.mode = str(dual_cfg.MODE).lower()
        self.prob = float(dual_cfg.PROB)
        self.direction = str(dual_cfg.DIRECTION).lower()
        self.pid_balanced = bool(dual_cfg.PID_BALANCED)
        self.crop_prob = float(dual_cfg.CROP_PROB)
        self.appearance_type = str(
            getattr(dual_cfg, 'APPEARANCE_TYPE', 'none')
        ).lower()
        self.appearance_target = str(
            getattr(dual_cfg, 'APPEARANCE_TARGET', 'mamba')
        ).lower()
        self.appearance_prob = float(
            getattr(dual_cfg, 'APPEARANCE_PROB', 0.0)
        )
        self.appearance_strength = float(
            getattr(dual_cfg, 'APPEARANCE_STRENGTH', 0.2)
        )
        if self.mode not in self._MODES:
            raise ValueError(
                'INPUT.DUAL_VIEW.MODE must be one of {}'.format(
                    sorted(self._MODES)
                )
            )
        if self.direction not in self._DIRECTIONS:
            raise ValueError(
                "INPUT.DUAL_VIEW.DIRECTION must be 'mamba_erased' or 'osnet_erased'"
            )
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError('INPUT.DUAL_VIEW.PROB must be in [0, 1]')
        if not 0.0 <= self.crop_prob <= 1.0:
            raise ValueError('INPUT.DUAL_VIEW.CROP_PROB must be in [0, 1]')
        if self.pid_balanced and self.mode != 'anticorr':
            raise ValueError(
                'INPUT.DUAL_VIEW.PID_BALANCED is only supported for anticorr mode'
            )
        if self.appearance_type not in self._APPEARANCE_TYPES:
            raise ValueError(
                'INPUT.DUAL_VIEW.APPEARANCE_TYPE must be one of {}'.format(
                    sorted(self._APPEARANCE_TYPES)
                )
            )
        if self.appearance_target not in self._APPEARANCE_TARGETS:
            raise ValueError(
                'INPUT.DUAL_VIEW.APPEARANCE_TARGET must be one of {}'.format(
                    sorted(self._APPEARANCE_TARGETS)
                )
            )
        if not 0.0 <= self.appearance_prob <= 1.0:
            raise ValueError(
                'INPUT.DUAL_VIEW.APPEARANCE_PROB must be in [0, 1]'
            )
        if self.appearance_strength < 0.0:
            raise ValueError(
                'INPUT.DUAL_VIEW.APPEARANCE_STRENGTH must be non-negative'
            )
        if (
            self.appearance_type in {'color', 'grayscale'}
            and self.appearance_strength > 1.0
        ):
            raise ValueError(
                'color/grayscale APPEARANCE_STRENGTH must be in [0, 1]'
            )
        self.pixel_mean = torch.tensor(
            cfg.INPUT.PIXEL_MEAN,
            dtype=torch.float32,
        ).view(3, 1, 1)
        self.pixel_std = torch.tensor(
            cfg.INPUT.PIXEL_STD,
            dtype=torch.float32,
        ).view(3, 1, 1)
        self.eraser = RandomErasing(
            probability=1.0,
            mode='pixel',
            max_count=1,
            device='cpu',
        )

    @staticmethod
    def _event(probability):
        return bool(torch.rand(()) < probability)

    def _erase(self, image):
        return self.eraser(image.clone())

    def _to_image_space(self, image):
        mean = self.pixel_mean.to(dtype=image.dtype, device=image.device)
        std = self.pixel_std.to(dtype=image.dtype, device=image.device)
        return (image * std + mean).clamp(0.0, 1.0)

    def _to_normalized_space(self, image):
        mean = self.pixel_mean.to(dtype=image.dtype, device=image.device)
        std = self.pixel_std.to(dtype=image.dtype, device=image.device)
        return (image.clamp(0.0, 1.0) - mean) / std

    def _appearance(self, image):
        if self.appearance_type == 'none':
            return image
        value = self._to_image_space(image)
        strength = self.appearance_strength
        if self.appearance_type == 'color':
            low = max(0.0, 1.0 - strength)
            high = 1.0 + strength
            factors = low + (high - low) * torch.rand(3)
            operations = (
                lambda x: TF.adjust_brightness(x, float(factors[0])),
                lambda x: TF.adjust_contrast(x, float(factors[1])),
                lambda x: TF.adjust_saturation(x, float(factors[2])),
            )
            for op_idx in torch.randperm(len(operations)).tolist():
                value = operations[op_idx](value)
        elif self.appearance_type == 'grayscale':
            gray = TF.rgb_to_grayscale(value, num_output_channels=3)
            value = value * (1.0 - strength) + gray * strength
        else:  # blur
            sigma = max(0.1, strength)
            kernel_size = max(3, 2 * int(round(2.0 * sigma)) + 1)
            value = TF.gaussian_blur(
                value,
                [kernel_size, kernel_size],
                [sigma, sigma],
            )
        return self._to_normalized_space(value)

    def _apply_appearance_pair(self, mamba_view, osnet_view):
        if (
            self.appearance_type == 'none'
            or not self._event(self.appearance_prob)
        ):
            return mamba_view, osnet_view
        target = self.appearance_target
        if target == 'random_one':
            target = 'mamba' if self._event(0.5) else 'osnet'
        if target == 'shared':
            # The shared routing mode supplies the same tensor to both
            # backbones. Reusing one transformed tensor also reuses the exact
            # sampled photometric parameters.
            shared_view = self._appearance(mamba_view)
            return shared_view, shared_view
        if target == 'mamba':
            return self._appearance(mamba_view), osnet_view
        return mamba_view, self._appearance(osnet_view)

    def _directional_pair(self, clean, direction):
        erased = self._erase(clean)
        if direction == 'mamba_erased':
            return erased, clean
        return clean, erased

    def _balanced_directions(self, pids):
        directions = {}
        pid_to_indices = {}
        for idx, pid in enumerate(pids):
            pid_to_indices.setdefault(int(pid), []).append(idx)
        for indices in pid_to_indices.values():
            permutation = torch.randperm(len(indices)).tolist()
            split = len(indices) // 2
            for order, perm_idx in enumerate(permutation):
                sample_idx = indices[perm_idx]
                directions[sample_idx] = (
                    'mamba_erased' if order < split else 'osnet_erased'
                )
        return directions

    def __call__(self, batch):
        cleans, crops, pids, camids, viewids, _ = zip(*batch)
        balanced = (
            self._balanced_directions(pids)
            if self.pid_balanced
            else None
        )
        mamba_views = []
        osnet_views = []

        for idx, (clean, crop) in enumerate(zip(cleans, crops)):
            if self.mode == 'shared':
                if self._event(self.prob):
                    erased = self._erase(clean)
                    mamba_view, osnet_view = erased, erased
                else:
                    mamba_view, osnet_view = clean, clean
            elif self.mode == 'diffmask':
                if self._event(self.prob):
                    mamba_view = self._erase(clean)
                    osnet_view = self._erase(clean)
                else:
                    mamba_view, osnet_view = clean, clean
            elif self.mode == 'independent':
                mamba_view = self._erase(clean) if self._event(self.prob) else clean
                osnet_view = self._erase(clean) if self._event(self.prob) else clean
            elif self.mode == 'anticorr':
                direction = (
                    balanced[idx]
                    if balanced is not None
                    else (
                        'mamba_erased'
                        if self._event(0.5)
                        else 'osnet_erased'
                    )
                )
                mamba_view, osnet_view = self._directional_pair(clean, direction)
            elif self.mode == 'fixed':
                mamba_view, osnet_view = self._directional_pair(
                    clean,
                    self.direction,
                )
            elif self.mode == 'anchor':
                if self._event(self.prob):
                    mamba_view, osnet_view = self._directional_pair(
                        clean,
                        self.direction,
                    )
                else:
                    mamba_view, osnet_view = clean, clean
            else:  # state_sample: same-compute clean/erase/crop state reference
                state = int(torch.randint(0, 3, ()).item())
                if state == 0:
                    mamba_view, osnet_view = clean, clean
                elif state == 1:
                    erased = self._erase(clean)
                    mamba_view, osnet_view = erased, erased
                else:
                    mamba_view, osnet_view = crop, crop

            if self.crop_prob > 0.0 and self._event(self.crop_prob):
                mamba_view, osnet_view = crop, crop

            mamba_view, osnet_view = self._apply_appearance_pair(
                mamba_view,
                osnet_view,
            )

            mamba_views.append(mamba_view)
            osnet_views.append(osnet_view)

        return (
            torch.stack(mamba_views, dim=0),
            torch.stack(osnet_views, dim=0),
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
        )


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    pam_enabled = cfg.INPUT.PAM.ENABLED
    dual_view_enabled = bool(getattr(cfg.INPUT.DUAL_VIEW, 'ENABLED', False))
    if pam_enabled and dual_view_enabled:
        raise ValueError('INPUT.PAM and INPUT.DUAL_VIEW are mutually exclusive')
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    if dual_view_enabled:
        dual_clean_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
        dual_crop_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.DUAL_VIEW.CROP_PADDING),
            T.RandomResizedCrop(
                cfg.INPUT.SIZE_TRAIN,
                scale=tuple(cfg.INPUT.DUAL_VIEW.CROP_SCALE),
                ratio=tuple(cfg.INPUT.DUAL_VIEW.CROP_RATIO),
                interpolation=3,
            ),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])

    if pam_enabled:
        pam_aug_mode = getattr(cfg.INPUT.PAM, 'AUG_MODE', 'default').lower()
        if pam_aug_mode == 'pade':
            pam_base_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ])
            pam_crop_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.Pad(cfg.INPUT.PAM.CROP_PADDING),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                T.RandomResizedCrop(
                    cfg.INPUT.SIZE_TRAIN,
                    scale=tuple(cfg.INPUT.PAM.CROP_SCALE),
                    ratio=tuple(cfg.INPUT.PAM.CROP_RATIO),
                    interpolation=3,
                ),
            ])
            pam_eraser_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=1.0, mode='pixel', max_count=1, device='cpu'),
            ])
        elif pam_aug_mode == 'default':
            pam_base_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ])
            pam_crop_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PAM.CROP_PADDING),
                T.RandomResizedCrop(
                    cfg.INPUT.SIZE_TRAIN,
                    scale=tuple(cfg.INPUT.PAM.CROP_SCALE),
                    ratio=tuple(cfg.INPUT.PAM.CROP_RATIO),
                    interpolation=3,
                ),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ])
            pam_eraser_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=1.0, mode='pixel', max_count=1, device='cpu'),
            ])
        else:
            raise ValueError(f'Unsupported INPUT.PAM.AUG_MODE: {pam_aug_mode}')

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    if pam_enabled:
        print(f'[Data] PAM enabled: BA + CA + EA training views ({pam_aug_mode} aug)')
        train_set = ParallelAugmentationImageDataset(
            dataset.train,
            pam_base_transforms,
            pam_crop_transforms,
            pam_eraser_transforms,
        )
        train_collate = pam_train_collate_fn
    elif dual_view_enabled:
        dual_cfg = cfg.INPUT.DUAL_VIEW
        print(
            '[Data] DUAL_VIEW enabled: mode={}, prob={:.3f}, direction={}, '
            'pid_balanced={}, crop_prob={:.3f}, appearance={}/{}/p{:.3f}/s{:.3f} '
            '(independent from PAM)'.format(
                str(dual_cfg.MODE).lower(),
                float(dual_cfg.PROB),
                str(dual_cfg.DIRECTION).lower(),
                bool(dual_cfg.PID_BALANCED),
                float(dual_cfg.CROP_PROB),
                str(getattr(dual_cfg, 'APPEARANCE_TYPE', 'none')).lower(),
                str(getattr(dual_cfg, 'APPEARANCE_TARGET', 'mamba')).lower(),
                float(getattr(dual_cfg, 'APPEARANCE_PROB', 0.0)),
                float(getattr(dual_cfg, 'APPEARANCE_STRENGTH', 0.2)),
            )
        )
        train_set = DualViewImageDataset(
            dataset.train,
            dual_clean_transforms,
            dual_crop_transforms,
        )
        train_collate = DualViewTrainCollator(cfg)
    else:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_collate = train_collate_fn
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate,
                pin_memory=True, persistent_workers=True if num_workers > 0 else False,
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, pin_memory=True,
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, pin_memory=True,
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
