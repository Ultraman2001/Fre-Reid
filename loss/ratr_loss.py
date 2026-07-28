"""
RATR (Ranking-aware Triplet Regularization) Loss
基于 ReIDMamba 实现，适配 2 分支场景（backbone + fused）

核心思想：
- ratr_intra_loss: 最小化正样本对距离排名的 Kendall-tau 相关性
- ratr_inter_loss: 最小化类中心距离排名的 Kendall-tau 相关性
- 强迫不同分支学习互补的判别特征

依赖条件：
- PK 采样 (sampler 保证 batch 按 ID 成块排列)
- 输入特征需要 L2 归一化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RATRIntraLoss(nn.Module):
    """
    RATR Intra-class Loss: 最小化正样本距离排名的相关性
    
    Args:
        num_branches: 分支数量 (N)，默认 2
        num_classes: PK 采样中的 P 值，默认 16
        samples_per_class: PK 采样中的 K 值，默认 4
        tau: Kendall-tau 温度参数，默认 0.1
    """
    def __init__(self, num_branches=2, num_classes=16, samples_per_class=4, tau=0.1):
        super().__init__()
        self.N = num_branches
        self.P = num_classes
        self.K = samples_per_class
        self.tau = tau
        
        # D = K - 1 (每个样本有 K-1 个正样本)
        D = self.K - 1
        self.D = D
        
        # 预计算 Kendall-tau 的索引对 (i < j)
        i = torch.arange(D).view(1, -1).expand(D, D)
        j = i.t()
        m = i < j
        self.register_buffer('pair_i', i[m].view(-1))
        self.register_buffer('pair_j', j[m].view(-1))
        
        # 预计算正样本索引 (假设 batch 按 ID 成块排列: [id0_k0, id0_k1, ..., id1_k0, ...])
        # targets 形状: [0,0,0,0,1,1,1,1,...] (P 个 ID，每个 K 个样本)
        targets = torch.arange(self.P).repeat_interleave(self.K)
        pos_idxs = targets.view(-1, 1) == targets.view(1, -1)  # (B, B)
        # 排除自身
        pos_idxs[torch.arange(self.P * self.K), torch.arange(self.P * self.K)] = False
        self.register_buffer('pos_idxs', pos_idxs)
        
    def _kendall_tau(self, x, y):
        """计算可微分的 Kendall-tau 相关系数"""
        # x, y: (B, D) - 每个样本对其正样本的距离
        concordant = torch.tanh((x[:, self.pair_i] - x[:, self.pair_j]) / self.tau) * \
                     torch.tanh((y[:, self.pair_i] - y[:, self.pair_j]) / self.tau)
        total_pairs = (self.D * (self.D - 1)) / 2.0
        return concordant.sum(-1).mean() / total_pairs
        
    def forward(self, feat_list, targets):
        """
        Args:
            feat_list: list of L2-normalized features, 每个形状 (B, D)
            targets: 标签 (B,)，实际上未使用（依赖预计算索引）
        
        Returns:
            intra_loss: 标量
        """
        B = targets.size(0)
        expected_B = self.P * self.K
        
        # Batch size 不匹配时跳过（例如最后一个不完整 batch）
        if B != expected_B:
            return torch.tensor(0.0, device=targets.device, dtype=feat_list[0].dtype)
        
        # 计算每个分支的正样本相似度
        pos_sims = []
        for feat in feat_list:
            # 相似度矩阵
            sim = torch.mm(feat, feat.t())  # (B, B)
            # 提取正样本相似度
            pos_sim = sim[self.pos_idxs].view(B, self.D)
            pos_sims.append(pos_sim)
        
        # 计算所有分支对的 Kendall-tau
        loss = 0.0
        count = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                loss = loss + self._kendall_tau(pos_sims[i], pos_sims[j])
                count += 1
        
        if count > 0:
            loss = loss / count
            
        return loss


class RATRInterLoss(nn.Module):
    """
    RATR Inter-class Loss: 最小化类中心距离排名的相关性
    
    Args:
        num_branches: 分支数量 (N)，默认 2
        num_classes: PK 采样中的 P 值，默认 16
        samples_per_class: PK 采样中的 K 值，默认 4
        tau: Kendall-tau 温度参数，默认 0.1
    """
    def __init__(self, num_branches=2, num_classes=16, samples_per_class=4, tau=0.1):
        super().__init__()
        self.N = num_branches
        self.P = num_classes
        self.K = samples_per_class
        self.tau = tau
        
        # D = P - 1 (每个样本有 P-1 个负类)
        D = self.P - 1
        self.D = D
        
        # 预计算 Kendall-tau 的索引对 (i < j)
        i = torch.arange(D).view(1, -1).expand(D, D)
        j = i.t()
        m = i < j
        self.register_buffer('pair_i', i[m].view(-1))
        self.register_buffer('pair_j', j[m].view(-1))
        
        # 预计算负类索引
        # targets 形状: [0,0,0,0,1,1,1,1,...] -> 每个样本对应的类 ID (0 到 P-1)
        targets = torch.arange(self.P).repeat_interleave(self.K)
        # neg_idxs: (B, P)，表示每个样本到各类中心的距离中哪些是负类
        neg_idxs = targets.view(-1, 1) != torch.arange(self.P).view(1, -1)
        self.register_buffer('neg_idxs', neg_idxs)
        
    def _kendall_tau(self, x, y):
        """计算可微分的 Kendall-tau 相关系数"""
        concordant = torch.tanh((x[:, self.pair_i] - x[:, self.pair_j]) / self.tau) * \
                     torch.tanh((y[:, self.pair_i] - y[:, self.pair_j]) / self.tau)
        total_pairs = (self.D * (self.D - 1)) / 2.0
        return concordant.sum(-1).mean() / total_pairs
        
    def forward(self, feat_list, targets):
        """
        Args:
            feat_list: list of L2-normalized features, 每个形状 (B, D)
            targets: 标签 (B,)，实际上未使用（依赖预计算索引）
        
        Returns:
            inter_loss: 标量
        """
        B = targets.size(0)
        expected_B = self.P * self.K
        
        # Batch size 不匹配时跳过
        if B != expected_B:
            return torch.tensor(0.0, device=targets.device, dtype=feat_list[0].dtype)
        
        # 计算每个分支的负类中心相似度
        neg_sims = []
        for feat in feat_list:
            # 计算类中心 (P, D)
            feat_reshaped = feat.view(self.P, self.K, -1)  # (P, K, feat_dim)
            centers = F.normalize(feat_reshaped.mean(dim=1), dim=1)  # (P, feat_dim)
            
            # 样本到各类中心的相似度 (B, P)
            sim_to_centers = torch.mm(feat, centers.t())
            
            # 提取负类相似度 (B, P-1)
            neg_sim = sim_to_centers[self.neg_idxs].view(B, self.D)
            neg_sims.append(neg_sim)
        
        # 计算所有分支对的 Kendall-tau
        loss = 0.0
        count = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                loss = loss + self._kendall_tau(neg_sims[i], neg_sims[j])
                count += 1
        
        if count > 0:
            loss = loss / count
            
        return loss


class RATRLoss(nn.Module):
    """
    RATR Loss: 结合 Intra 和 Inter 损失
    
    Args:
        num_branches: 分支数量，默认 2 (backbone + fused)
        num_classes: PK 采样中的 P 值，默认 16
        samples_per_class: PK 采样中的 K 值，默认 4
        tau: Kendall-tau 温度参数，默认 0.1
    """
    def __init__(
        self,
        num_branches=2,
        num_classes=16,
        samples_per_class=4,
        tau=0.1,
        mode='raw',
        intra_target=0.5,
        inter_target=0.3,
    ):
        super().__init__()
        self.intra_loss = RATRIntraLoss(num_branches, num_classes, samples_per_class, tau)
        self.inter_loss = RATRInterLoss(num_branches, num_classes, samples_per_class, tau)
        self.mode = str(mode).lower()
        if self.mode not in ('raw', 'hinge', 'square'):
            raise ValueError('RATR_MODE must be one of: raw, hinge, square')
        self.intra_target = float(intra_target)
        self.inter_target = float(inter_target)
        self.last_intra = 0.0
        self.last_inter = 0.0

        print(
            '[RATR] Initialized: N={}, P={}, K={}, tau={}, mode={}, targets={}/{}'.format(
                num_branches,
                num_classes,
                samples_per_class,
                tau,
                self.mode,
                self.intra_target,
                self.inter_target,
            )
        )
        
    def forward(self, feat_list, targets):
        """
        Args:
            feat_list: [backbone_feat, fused_feat]，均已 L2 归一化
            targets: 标签 (B,)
        
        Returns:
            ratr_loss: intra_loss + inter_loss
        """
        intra = self.intra_loss(feat_list, targets)
        inter = self.inter_loss(feat_list, targets)
        self.last_intra = intra.detach().item()
        self.last_inter = inter.detach().item()
        if self.mode == 'hinge':
            return F.relu(intra - self.intra_target) + F.relu(inter - self.inter_target)
        if self.mode == 'square':
            return intra.square() + inter.square()
        return intra + inter
