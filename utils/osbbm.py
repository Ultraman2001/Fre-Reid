import math

import torch


def _norm_stats(mean, std, device, dtype):
    mean = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


def _different_identity_permutation(labels):
    batch_size = labels.shape[0]
    if batch_size <= 1:
        return torch.arange(batch_size, device=labels.device)

    perm = torch.empty(batch_size, device=labels.device, dtype=torch.long)
    all_indices = torch.arange(batch_size, device=labels.device)
    for idx in range(batch_size):
        candidates = all_indices[labels != labels[idx]]
        if candidates.numel() == 0:
            perm[idx] = idx
        else:
            choice = torch.randint(0, candidates.numel(), (), device=labels.device)
            perm[idx] = candidates[choice]
    return perm


def _different_identity_derangement(labels, attempts=64):
    """Build a one-to-one donor assignment whenever the batch permits it."""
    batch_size = labels.shape[0]
    identity = torch.arange(batch_size, device=labels.device)
    if batch_size <= 1:
        return identity

    _, counts = labels.unique(return_counts=True)
    max_count = int(counts.max().item())
    if max_count * 2 <= batch_size:
        order = torch.argsort(labels)
        perm = torch.empty_like(order)
        perm[order] = torch.roll(order, shifts=-max_count)
        if torch.all(labels[perm] != labels):
            return perm

    for _ in range(attempts):
        perm = torch.randperm(batch_size, device=labels.device)
        if torch.all(labels[perm] != labels):
            return perm
    return _different_identity_permutation(labels)


def _sample_rotated_rect(height, width, device, scale=(0.02, 0.25), ratio=(0.3, 3.3), attempts=10):
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(attempts):
        target_area = area * torch.empty((), device=device).uniform_(scale[0], scale[1]).item()
        aspect = math.exp(torch.empty((), device=device).uniform_(log_ratio[0], log_ratio[1]).item())
        rect_h = int(round(math.sqrt(target_area * aspect)))
        rect_w = int(round(math.sqrt(target_area / aspect)))
        if 0 < rect_h <= height and 0 < rect_w <= width:
            top = int(torch.randint(0, height - rect_h + 1, (), device=device).item())
            left = int(torch.randint(0, width - rect_w + 1, (), device=device).item())
            angle = torch.empty((), device=device).uniform_(0.0, 2.0 * math.pi).item()
            return top, left, rect_h, rect_w, angle
    rect_h = max(1, height // 4)
    rect_w = max(1, width // 4)
    top = int(torch.randint(0, height - rect_h + 1, (), device=device).item())
    left = int(torch.randint(0, width - rect_w + 1, (), device=device).item())
    angle = torch.empty((), device=device).uniform_(0.0, 2.0 * math.pi).item()
    return top, left, rect_h, rect_w, angle


def _rotated_rect_mask(height, width, top, left, rect_h, rect_w, angle, device):
    ys = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    cy = float(top) + (float(rect_h) - 1.0) * 0.5
    cx = float(left) + (float(rect_w) - 1.0) * 0.5
    x0 = xs - cx
    y0 = ys - cy
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xr = cos_a * x0 + sin_a * y0
    yr = -sin_a * x0 + cos_a * y0
    return (xr.abs() <= float(rect_w) * 0.5) & (yr.abs() <= float(rect_h) * 0.5)


def _apply_local_rotated_grayscale(images, prob, apply_mask=None):
    """Apply the local grayscale preprocessing described by OSBBM.

    For each selected image, a random rotated rectangle is converted to grayscale.
    The grayscale value either fills all RGB channels or only one random channel.
    """
    if prob <= 0 or images.shape[1] != 3:
        return images

    out = images.clone()
    batch_size, _, height, width = out.shape
    for idx in range(batch_size):
        if apply_mask is not None and not bool(apply_mask[idx]):
            continue
        if torch.rand((), device=out.device) >= prob:
            continue
        top, left, rect_h, rect_w, angle = _sample_rotated_rect(height, width, out.device)
        mask = _rotated_rect_mask(height, width, top, left, rect_h, rect_w, angle, out.device)
        if not mask.any():
            continue
        gray = out[idx].mean(dim=0)
        if torch.rand((), device=out.device) < 0.5:
            out[idx, :, mask] = gray[mask].unsqueeze(0).expand(3, -1)
        else:
            channel = int(torch.randint(0, 3, (), device=out.device).item())
            out[idx, channel, mask] = gray[mask]
    return out


def _sample_apply_mask(labels, prob, sample_mode):
    if sample_mode == 'random':
        return torch.rand(labels.shape[0], device=labels.device) < float(prob)
    if sample_mode != 'pk_half':
        raise ValueError("OSBBM SAMPLE_MODE must be 'random' or 'pk_half'")

    apply_mask = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)
    for identity in labels.unique():
        indices = torch.nonzero(labels == identity, as_tuple=False).flatten()
        count = int(round(indices.numel() * float(prob)))
        count = max(0, min(count, indices.numel()))
        if count > 0:
            selected = indices[torch.randperm(indices.numel(), device=labels.device)[:count]]
            apply_mask[selected] = True
    return apply_mask


def _sample_blocks(num_blocks, num_mix_blocks, block_mode, device):
    if block_mode == 'random':
        return torch.randperm(num_blocks, device=device)[:num_mix_blocks]
    if block_mode != 'part_balanced':
        raise ValueError("OSBBM BLOCK_MODE must be 'random' or 'part_balanced'")

    zones = torch.tensor_split(torch.arange(num_blocks, device=device), num_mix_blocks)
    selected = [zone[torch.randint(zone.numel(), (), device=device)] for zone in zones if zone.numel()]
    return torch.stack(selected)


def _build_mix_info(
    labels,
    prob,
    num_blocks,
    num_mix_blocks,
    sample_mode='random',
    donor_mode='random',
    block_mode='random',
):
    batch_size = labels.shape[0]
    if donor_mode == 'random':
        donor_perm = _different_identity_permutation(labels)
    elif donor_mode == 'derangement':
        donor_perm = _different_identity_derangement(labels)
    else:
        raise ValueError("OSBBM DONOR_MODE must be 'random' or 'derangement'")
    apply_mask = _sample_apply_mask(labels, prob, sample_mode)
    block_mask = torch.zeros(batch_size, num_blocks, device=labels.device, dtype=torch.bool)
    lam = torch.ones(batch_size, device=labels.device, dtype=torch.float32)

    for idx in torch.nonzero(apply_mask, as_tuple=False).flatten().tolist():
        donor_blocks = _sample_blocks(
            num_blocks,
            num_mix_blocks,
            block_mode,
            labels.device,
        )
        block_mask[idx, donor_blocks] = True
        lam[idx] = 1.0 - float(num_mix_blocks) / float(num_blocks)

    return {
        'donor_perm': donor_perm,
        'apply_mask': apply_mask,
        'block_mask': block_mask,
        'target_b': labels[donor_perm],
        'lambda': lam,
    }


def apply_osbbm_batch(
    images,
    labels,
    mean,
    std,
    prob=0.5,
    num_blocks=8,
    num_mix_blocks=2,
    gray_prob=0.5,
    gray_scope='all',
    sample_mode='random',
    donor_mode='random',
    block_mode='random',
    mix_info=None,
    return_info=False,
):
    """Apply OSBBM horizontal block mixing to a normalized image batch.

    Returns mixed images by default. When ``return_info`` is true, also returns
    the donor labels and per-sample lambda for OSBBM's mixed-label loss.
    """
    if prob <= 0 or images.shape[0] <= 1:
        if return_info:
            lam = torch.ones(images.shape[0], device=images.device, dtype=torch.float32)
            return images, labels.detach(), lam, None
        return images

    batch_size, _, height, _ = images.shape
    num_blocks = max(1, int(num_blocks))
    num_mix_blocks = max(1, min(int(num_mix_blocks), num_blocks))
    gray_prob = float(gray_prob)
    gray_scope = str(gray_scope).lower()
    if gray_scope not in ('all', 'mixed'):
        raise ValueError("OSBBM GRAY_SCOPE must be 'all' or 'mixed'")

    orig_dtype = images.dtype
    work = images.float()
    mean_t, std_t = _norm_stats(mean, std, work.device, work.dtype)
    work = (work * std_t + mean_t).clamp(0.0, 1.0)

    labels = labels.detach()
    if mix_info is None:
        mix_info = _build_mix_info(
            labels,
            prob,
            num_blocks,
            num_mix_blocks,
            sample_mode=sample_mode,
            donor_mode=donor_mode,
            block_mode=block_mode,
        )
    gray_mask = mix_info['apply_mask'] if gray_scope == 'mixed' else None
    work = _apply_local_rotated_grayscale(work, gray_prob, apply_mask=gray_mask)
    mixed = work.clone()
    donor_perm = mix_info['donor_perm'].to(device=work.device)
    block_mask = mix_info['block_mask'].to(device=work.device)
    target_b = mix_info['target_b'].to(device=labels.device)
    lam = mix_info['lambda'].to(device=labels.device, dtype=torch.float32)
    block_edges = [round(i * height / num_blocks) for i in range(num_blocks + 1)]

    for idx in range(batch_size):
        donor_idx = int(donor_perm[idx].item())
        block_ids = torch.nonzero(block_mask[idx], as_tuple=False).flatten().tolist()
        for block_id in block_ids:
            y0, y1 = block_edges[block_id], block_edges[block_id + 1]
            if y1 <= y0:
                continue
            mixed[idx, :, y0:y1, :] = work[donor_idx, :, y0:y1, :]

    mixed = (mixed - mean_t) / std_t
    mixed = mixed.to(orig_dtype)
    if return_info:
        return mixed, target_b, lam, mix_info
    return mixed
