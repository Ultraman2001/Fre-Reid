import torch


def _norm_stats(mean, std, device, dtype):
    mean = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


def _different_identity_permutation(labels):
    batch_size = labels.shape[0]
    perm = torch.randperm(batch_size, device=labels.device)
    if batch_size <= 1:
        return perm

    for _ in range(8):
        same = labels[perm] == labels
        if not same.any():
            break
        alt = torch.randperm(batch_size, device=labels.device)
        perm = torch.where(same, alt, perm)
    return perm


def apply_osbbm_batch(
    images,
    labels,
    mean,
    std,
    prob=0.5,
    num_blocks=8,
    num_mix_blocks=4,
    gray_prob=0.5,
):
    """Apply OSBBM-style horizontal block replacement to a normalized batch.

    Donor blocks are treated as occluders, so labels are kept unchanged. This
    is the variant used by the PAM + scheduled OSBBM experiments.
    """
    if prob <= 0 or images.shape[0] <= 1:
        return images

    _, channels, height, _ = images.shape
    num_blocks = max(1, int(num_blocks))
    num_mix_blocks = max(1, min(int(num_mix_blocks), num_blocks))
    gray_prob = float(gray_prob)

    orig_dtype = images.dtype
    work = images.float()
    mean_t, std_t = _norm_stats(mean, std, work.device, work.dtype)
    work = (work * std_t + mean_t).clamp(0.0, 1.0)
    mixed = work.clone()

    labels = labels.detach()
    donor_perm = _different_identity_permutation(labels)
    apply_mask = torch.rand(images.shape[0], device=work.device) < float(prob)
    block_edges = [round(i * height / num_blocks) for i in range(num_blocks + 1)]

    for idx in torch.nonzero(apply_mask, as_tuple=False).flatten().tolist():
        block_ids = torch.randperm(num_blocks, device=work.device)[:num_mix_blocks].tolist()
        donor_idx = int(donor_perm[idx].item())
        for block_id in block_ids:
            y0, y1 = block_edges[block_id], block_edges[block_id + 1]
            if y1 <= y0:
                continue
            patch = work[donor_idx, :, y0:y1, :].clone()
            if channels == 3 and gray_prob > 0 and torch.rand((), device=work.device) < gray_prob:
                patch = patch.mean(dim=0, keepdim=True).expand_as(patch)
            mixed[idx, :, y0:y1, :] = patch

    mixed = (mixed - mean_t) / std_t
    return mixed.to(orig_dtype)
