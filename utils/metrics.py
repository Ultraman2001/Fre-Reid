import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def body_part_distance(qf, gf, q_visibility=None, g_visibility=None, batch_size=1024):
    qf = qf.float()
    gf = gf.float()
    q_visibility = torch.ones(qf.shape[:2], dtype=torch.bool) if q_visibility is None else q_visibility
    g_visibility = torch.ones(gf.shape[:2], dtype=torch.bool) if g_visibility is None else g_visibility

    dist_chunks = []
    for start in range(0, gf.size(0), batch_size):
        end = min(start + batch_size, gf.size(0))
        gf_batch = gf[start:end]
        gv_batch = g_visibility[start:end]

        q_sq = (qf ** 2).sum(dim=2).unsqueeze(1)
        g_sq = (gf_batch ** 2).sum(dim=2).unsqueeze(0)
        prod = torch.einsum('qmd,gmd->qgm', qf, gf_batch)
        part_dist = (q_sq + g_sq - 2 * prod).clamp_min(0.0).sqrt()

        if q_visibility.dtype is torch.bool and gv_batch.dtype is torch.bool:
            valid = q_visibility.unsqueeze(1) & gv_batch.unsqueeze(0)
            weights = valid.float()
        else:
            weights = torch.sqrt(q_visibility.float().unsqueeze(1) * gv_batch.float().unsqueeze(0))
            valid = weights > 0

        denom = weights.sum(dim=2)
        pair_dist = (part_dist * weights).sum(dim=2) / denom.clamp_min(1e-6)
        pair_dist = pair_dist.masked_fill(denom <= 0, float('inf'))
        dist_chunks.append(pair_dist.cpu())

    distmat = torch.cat(dist_chunks, dim=1)
    finite = torch.isfinite(distmat)
    if finite.any():
        fill_value = distmat[finite].max() + 1.0
        distmat = distmat.masked_fill(~finite, fill_value)
    return distmat.numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, bp_anchor_distance_weight=0.0):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.bp_anchor_distance_weight = float(bp_anchor_distance_weight)
        if not 0.0 <= self.bp_anchor_distance_weight <= 1.0:
            raise ValueError("BPBreID anchor distance weight must be inside [0, 1]")

    def reset(self):
        self.feats = []
        self.bp_feats = []
        self.bp_visibility = []
        self.bp_anchor_feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        if isinstance(feat, dict) and 'bp_features' in feat:
            self.bp_feats.append(feat['bp_features'].cpu())
            self.bp_visibility.append(feat['bp_visibility'].cpu())
            self.feats.append(feat['concat'].cpu())
            if 'bp_anchor' in feat:
                self.bp_anchor_feats.append(feat['bp_anchor'].cpu())
        else:
            self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        if len(self.bp_feats) > 0:
            feats = torch.cat(self.bp_feats, dim=0)
            visibility = torch.cat(self.bp_visibility, dim=0)
            if self.feat_norm:
                print("The BPBreID part features are normalized")
                feats = torch.nn.functional.normalize(feats, dim=2, p=2)
            qf = feats[:self.num_query]
            gf = feats[self.num_query:]
            q_visibility = visibility[:self.num_query]
            g_visibility = visibility[self.num_query:]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            if self.reranking:
                print('=> BPBreID visibility distance does not support reranking; using part distance')
            print('=> Computing DistMat with BPBreID visibility-aware body-part distance')
            distmat = body_part_distance(qf, gf, q_visibility, g_visibility)
            if len(self.bp_anchor_feats) > 0:
                anchor_feats = torch.cat(self.bp_anchor_feats, dim=0)
                if self.feat_norm:
                    print("The BPBreID GeM anchor features are normalized")
                    anchor_feats = torch.nn.functional.normalize(anchor_feats, dim=1, p=2)
                q_anchor = anchor_feats[:self.num_query]
                g_anchor = anchor_feats[self.num_query:]
                anchor_distmat = np.sqrt(np.maximum(euclidean_distance(q_anchor, g_anchor), 0.0))
                bp_cmc, bp_map = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)
                anchor_cmc, anchor_map = eval_func(
                    anchor_distmat,
                    q_pids,
                    g_pids,
                    q_camids,
                    g_camids,
                    self.max_rank,
                )
                print(
                    "=> BPBreID diagnostics: BP-only mAP {:.1%}, Rank-1 {:.1%}; "
                    "anchor-only mAP {:.1%}, Rank-1 {:.1%}".format(
                        bp_map,
                        bp_cmc[0],
                        anchor_map,
                        anchor_cmc[0],
                    )
                )
                print(
                    "=> Fusing BPBreID and GeM anchor distances with anchor weight {:.2f}".format(
                        self.bp_anchor_distance_weight
                    )
                )
                distmat = (
                    (1.0 - self.bp_anchor_distance_weight) * distmat
                    + self.bp_anchor_distance_weight * anchor_distmat
                )
        else:
            feats = torch.cat(self.feats, dim=0)
            if self.feat_norm:
                print("The test feature is normalized")
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
            # query
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])

            g_camids = np.asarray(self.camids[self.num_query:])
            if self.reranking:
                print('=> Enter reranking')
                # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
                distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

            else:
                print('=> Computing DistMat with euclidean_distance')
                distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
