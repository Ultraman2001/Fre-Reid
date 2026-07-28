import json
import os
from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn.functional as F


_BRANCH_KEYS = OrderedDict([
    ('mamba', 'backbone'),
    ('fdmf', 'fdmf_only'),
    ('osnet', 'osnet'),
    ('stage3local', 'stage3_stripe_local'),
])


def _sample_indices(total, limit, seed):
    if total <= 0:
        return torch.empty(0, dtype=torch.long)
    if limit <= 0 or limit >= total:
        return torch.arange(total, dtype=torch.long)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:limit]


def _linear_cka(x, y):
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    cross = x.transpose(0, 1).matmul(y)
    xx = x.transpose(0, 1).matmul(x)
    yy = y.transpose(0, 1).matmul(y)
    numerator = cross.square().sum()
    denominator = torch.sqrt(xx.square().sum() * yy.square().sum()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _row_ranks(values):
    order = torch.argsort(values, dim=1)
    ranks = torch.empty_like(values, dtype=torch.float32)
    base = torch.arange(values.shape[1], dtype=torch.float32).view(1, -1)
    ranks.scatter_(1, order, base.expand(values.shape[0], -1))
    return ranks


def _mean_row_pearson(x, y):
    x = x.float() - x.float().mean(dim=1, keepdim=True)
    y = y.float() - y.float().mean(dim=1, keepdim=True)
    denominator = torch.sqrt(
        x.square().sum(dim=1) * y.square().sum(dim=1)
    ).clamp_min(1e-12)
    return float(((x * y).sum(dim=1) / denominator).mean().item())


def _retrieval_similarity(features, query_indices, gallery_indices):
    features = F.normalize(features.float(), p=2, dim=1)
    query = features.index_select(0, query_indices)
    gallery = features.index_select(0, gallery_indices)
    return query.matmul(gallery.transpose(0, 1))


def _topk_indices(similarity, topk):
    topk = min(int(topk), similarity.shape[1])
    if topk <= 0:
        return torch.empty(similarity.shape[0], 0, dtype=torch.long)
    return similarity.topk(topk, dim=1, largest=True).indices


def _topk_overlap(idx_a, idx_b):
    topk = min(idx_a.shape[1], idx_b.shape[1])
    if topk <= 0:
        return 0.0
    idx_a = idx_a[:, :topk]
    idx_b = idx_b[:, :topk]
    overlap = (
        idx_a.unsqueeze(2) == idx_b.unsqueeze(1)
    ).any(dim=2).float().sum(dim=1) / float(topk)
    return float(overlap.mean().item())


def get_branch_features(feature_tensors):
    branches = OrderedDict()
    for branch_name, feature_key in _BRANCH_KEYS.items():
        if feature_key in feature_tensors:
            branches[branch_name] = feature_tensors[feature_key]
    return branches


def build_branch_normalized_descriptors(feature_tensors):
    branches = get_branch_features(feature_tensors)
    descriptors = OrderedDict()
    for name, feature in branches.items():
        descriptors['bnorm_' + name] = F.normalize(feature.float(), p=2, dim=1)

    base_names = [name for name in ('mamba', 'fdmf', 'osnet') if name in branches]
    for size in (2, 3):
        for names in combinations(base_names, size):
            key = 'bnorm_' + '_'.join(names)
            descriptors[key] = torch.cat(
                [F.normalize(branches[name].float(), p=2, dim=1) for name in names],
                dim=1,
            )

    if all(name in branches for name in ('mamba', 'fdmf', 'osnet', 'stage3local')):
        descriptors['bnorm_mamba_fdmf_osnet_stage3local'] = torch.cat(
            [
                F.normalize(branches['mamba'].float(), p=2, dim=1),
                F.normalize(branches['fdmf'].float(), p=2, dim=1),
                F.normalize(branches['osnet'].float(), p=2, dim=1),
                F.normalize(branches['stage3local'].float(), p=2, dim=1),
            ],
            dim=1,
        )
    return descriptors


def _weight_tag(value):
    return '{:03d}'.format(int(round(100.0 * float(value))))


def iter_weighted_branch_descriptors(
    feature_tensors,
    fdmf_weights,
    osnet_weights,
    local_weights,
):
    branches = get_branch_features(feature_tensors)
    required = ('mamba', 'fdmf', 'osnet')
    if not all(name in branches for name in required):
        return

    normalized = {
        name: F.normalize(feature.float(), p=2, dim=1)
        for name, feature in branches.items()
    }
    has_local = 'stage3local' in normalized
    scan_local_weights = list(local_weights) if has_local else [None]

    for fdmf_weight in fdmf_weights:
        for osnet_weight in osnet_weights:
            for local_weight in scan_local_weights:
                weights = {
                    'mamba': 1.0,
                    'fdmf': float(fdmf_weight),
                    'osnet': float(osnet_weight),
                }
                parts = [normalized['mamba']]
                if float(fdmf_weight) != 0.0:
                    parts.append(float(fdmf_weight) * normalized['fdmf'])
                if float(osnet_weight) != 0.0:
                    parts.append(float(osnet_weight) * normalized['osnet'])

                key = 'ws_m100_f{}_o{}'.format(
                    _weight_tag(fdmf_weight),
                    _weight_tag(osnet_weight),
                )
                if has_local:
                    local_weight = float(local_weight)
                    weights['stage3local'] = local_weight
                    key += '_l{}'.format(_weight_tag(local_weight))
                    if local_weight != 0.0:
                        parts.append(local_weight * normalized['stage3local'])

                yield key, torch.cat(parts, dim=1), weights


def analyze_feature_complementarity(
    feature_tensors,
    num_query,
    topk=10,
    cka_samples=2048,
    query_samples=256,
    gallery_samples=4096,
    seed=42,
):
    branches = get_branch_features(feature_tensors)
    total = next(iter(branches.values())).shape[0] if branches else 0
    gallery_count = max(total - int(num_query), 0)
    cka_idx = _sample_indices(total, int(cka_samples), seed)
    query_idx = _sample_indices(int(num_query), int(query_samples), seed + 1)
    gallery_idx = _sample_indices(gallery_count, int(gallery_samples), seed + 2) + int(num_query)

    report = {
        'sampling': {
            'total': int(total),
            'num_query': int(num_query),
            'cka_samples': int(cka_idx.numel()),
            'query_samples': int(query_idx.numel()),
            'gallery_samples': int(gallery_idx.numel()),
            'topk': int(topk),
            'seed': int(seed),
        },
        'branches': {},
        'pairwise': {},
    }

    similarities = {}
    for name, feature in branches.items():
        norms = torch.norm(feature.float(), p=2, dim=1)
        report['branches'][name] = {
            'dim': int(feature.shape[1]),
            'norm_mean': float(norms.mean().item()),
            'norm_std': float(norms.std(unbiased=False).item()),
        }
        if query_idx.numel() > 0 and gallery_idx.numel() > 0:
            similarities[name] = _retrieval_similarity(feature, query_idx, gallery_idx)

    distance_ranks = {
        name: _row_ranks(1.0 - similarity)
        for name, similarity in similarities.items()
    }
    neighbor_indices = {
        name: _topk_indices(similarity, topk)
        for name, similarity in similarities.items()
    }

    for name_a, name_b in combinations(branches.keys(), 2):
        feature_a = branches[name_a].index_select(0, cka_idx)
        feature_b = branches[name_b].index_select(0, cka_idx)
        pair_report = {
            'linear_cka': _linear_cka(feature_a, feature_b),
        }
        if name_a in similarities and name_b in similarities:
            pair_report['distance_spearman'] = _mean_row_pearson(
                distance_ranks[name_a],
                distance_ranks[name_b],
            )
            pair_report['topk_overlap'] = _topk_overlap(
                neighbor_indices[name_a],
                neighbor_indices[name_b],
            )
        report['pairwise'][name_a + '__' + name_b] = pair_report
    return report


def attach_retrieval_results(report, results):
    retrieval = {}
    for name, result in results.items():
        cmc = result['cmc']
        retrieval[name] = {
            'mAP': float(result['mAP']),
            'R1': float(cmc[0]) if len(cmc) > 0 else 0.0,
            'R5': float(cmc[4]) if len(cmc) > 4 else 0.0,
            'R10': float(cmc[9]) if len(cmc) > 9 else 0.0,
        }
    report['retrieval'] = retrieval

    comparisons = {
        'fdmf_given_mamba_osnet': ('bnorm_mamba_fdmf_osnet', 'bnorm_mamba_osnet'),
        'mamba_given_fdmf_osnet': ('bnorm_mamba_fdmf_osnet', 'bnorm_fdmf_osnet'),
        'osnet_given_mamba_fdmf': ('bnorm_mamba_fdmf_osnet', 'bnorm_mamba_fdmf'),
        'stage3local_given_mamba_fdmf_osnet': (
            'bnorm_mamba_fdmf_osnet_stage3local',
            'bnorm_mamba_fdmf_osnet',
        ),
    }
    marginal = {}
    for label, (with_key, without_key) in comparisons.items():
        if with_key not in retrieval or without_key not in retrieval:
            continue
        marginal[label] = {
            'delta_mAP': retrieval[with_key]['mAP'] - retrieval[without_key]['mAP'],
            'delta_R1': retrieval[with_key]['R1'] - retrieval[without_key]['R1'],
        }
    report['marginal_gain'] = marginal
    return report


def attach_weight_sweep_results(report, weight_metadata, results):
    weighted_results = {}
    for name, weights in weight_metadata.items():
        if name not in results:
            continue
        result = results[name]
        cmc = result['cmc']
        weighted_results[name] = {
            'weights': weights,
            'mAP': float(result['mAP']),
            'R1': float(cmc[0]) if len(cmc) > 0 else 0.0,
            'R5': float(cmc[4]) if len(cmc) > 4 else 0.0,
            'R10': float(cmc[9]) if len(cmc) > 9 else 0.0,
        }
    report['weight_sweep'] = weighted_results
    return report


def log_feature_complementarity(logger, report):
    logger.info('=== BRANCH COMPLEMENTARITY ===')
    for name, stats in report.get('branches', {}).items():
        logger.info(
            'Branch {:<12} dim={:<5d} norm={:.4f}+/-{:.4f}'.format(
                name,
                stats['dim'],
                stats['norm_mean'],
                stats['norm_std'],
            )
        )
    for name, stats in report.get('pairwise', {}).items():
        logger.info(
            'Pair {:<28} CKA={:.4f} dist_rho={:.4f} topk_overlap={:.4f}'.format(
                name,
                stats['linear_cka'],
                stats.get('distance_spearman', 0.0),
                stats.get('topk_overlap', 0.0),
            )
        )
    for name, stats in report.get('marginal_gain', {}).items():
        logger.info(
            'Marginal {:<36} delta_mAP={:+.2f}pp delta_R1={:+.2f}pp'.format(
                name,
                100.0 * stats['delta_mAP'],
                100.0 * stats['delta_R1'],
            )
        )


def save_feature_complementarity(report, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
