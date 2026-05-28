"""
Gate Monitor - 用于监控门控Context Mixer的学习情况
"""
import torch
import numpy as np
from collections import defaultdict
import os
import json


class GateMonitor:
    """监控Gate统计信息的工具类"""
    
    def __init__(self, log_dir='./logs', log_interval=100, save_stats=True):
        """
        Args:
            log_dir: 日志保存目录
            log_interval: 打印间隔（每N个batch打印一次）
            save_stats: 是否保存统计信息到文件
        """
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.save_stats = save_stats
        
        # 统计信息存储
        self.stats_history = defaultdict(list)
        self.batch_counter = 0
        self.epoch_counter = 0
        
        # 创建日志目录
        if self.save_stats:
            os.makedirs(log_dir, exist_ok=True)
    
    def update(self, gate, stage_name='unknown'):
        """
        更新Gate统计
        
        Args:
            gate: Tensor (B, C, H, W) 门控权重
            stage_name: Stage名称（如'stage2', 'stage3'）
        """
        with torch.no_grad():
            gate_mean = gate.mean().item()
            gate_std = gate.std().item()
            gate_min = gate.min().item()
            gate_max = gate.max().item()
            gate_median = gate.median().item()
            
            # 计算分布（有多少gate接近0或1）
            near_zero = (gate < 0.1).float().mean().item()  # <0.1的比例
            near_one = (gate > 0.9).float().mean().item()   # >0.9的比例
            mid_range = ((gate >= 0.3) & (gate <= 0.7)).float().mean().item()  # [0.3, 0.7]的比例
            
            stats = {
                'mean': gate_mean,
                'std': gate_std,
                'min': gate_min,
                'max': gate_max,
                'median': gate_median,
                'near_zero': near_zero,
                'near_one': near_one,
                'mid_range': mid_range,
                'batch': self.batch_counter,
                'epoch': self.epoch_counter,
            }
            
            # 保存历史
            for key, value in stats.items():
                self.stats_history[f'{stage_name}_{key}'].append(value)
            
            # 打印日志
            if self.batch_counter % self.log_interval == 0:
                print(f"[{stage_name}] Batch {self.batch_counter} | "
                      f"Gate: mean={gate_mean:.4f}, std={gate_std:.4f}, "
                      f"median={gate_median:.4f}, range=[{gate_min:.4f}, {gate_max:.4f}] | "
                      f"Distribution: <0.1={near_zero:.2%}, 0.3~0.7={mid_range:.2%}, >0.9={near_one:.2%}")
            
            self.batch_counter += 1
            
            return stats
    
    def end_epoch(self, epoch):
        """Epoch结束时调用"""
        self.epoch_counter = epoch
        
        if self.save_stats:
            # 保存统计信息
            stats_file = os.path.join(self.log_dir, f'gate_stats_epoch_{epoch}.json')
            
            # 只保存当前epoch的统计
            epoch_stats = {}
            for key, values in self.stats_history.items():
                if len(values) > 0:
                    epoch_stats[key] = values[-1] if isinstance(values[-1], (int, float)) else float(values[-1])
            
            with open(stats_file, 'w') as f:
                json.dump(epoch_stats, f, indent=2)
            
            print(f"\n[Gate Monitor] Epoch {epoch} stats saved to {stats_file}")
    
    def summarize(self):
        """总结所有统计信息"""
        print("\n" + "="*80)
        print("Gate Monitor Summary")
        print("="*80)
        
        for key in ['stage2_mean', 'stage2_std', 'stage3_mean', 'stage3_std']:
            if key in self.stats_history and len(self.stats_history[key]) > 0:
                values = self.stats_history[key]
                print(f"{key:20s}: init={values[0]:.4f}, final={values[-1]:.4f}, "
                      f"avg={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        print("="*80 + "\n")
    
    def plot_stats(self, save_path=None):
        """绘制Gate统计曲线（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Mean over time
        for stage in ['stage2', 'stage3']:
            key = f'{stage}_mean'
            if key in self.stats_history:
                axes[0, 0].plot(self.stats_history[key], label=stage)
        axes[0, 0].set_title('Gate Mean over Training')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Std over time
        for stage in ['stage2', 'stage3']:
            key = f'{stage}_std'
            if key in self.stats_history:
                axes[0, 1].plot(self.stats_history[key], label=stage)
        axes[0, 1].set_title('Gate Std over Training')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution (near zero/one)
        for stage in ['stage2', 'stage3']:
            key_zero = f'{stage}_near_zero'
            key_one = f'{stage}_near_one'
            if key_zero in self.stats_history:
                axes[1, 0].plot(self.stats_history[key_zero], label=f'{stage} <0.1', linestyle='--')
                axes[1, 0].plot(self.stats_history[key_one], label=f'{stage} >0.9', linestyle='-')
        axes[1, 0].set_title('Gate Distribution Extremes')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Median
        for stage in ['stage2', 'stage3']:
            key = f'{stage}_median'
            if key in self.stats_history:
                axes[1, 1].plot(self.stats_history[key], label=stage)
        axes[1, 1].set_title('Gate Median over Training')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Median')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gate statistics plot saved to {save_path}")
        else:
            plt.show()


# 简单的使用示例
if __name__ == '__main__':
    # 模拟使用
    monitor = GateMonitor(log_dir='./logs/gate_monitor', log_interval=10)
    
    # 模拟训练过程
    for epoch in range(3):
        for batch in range(100):
            # 模拟gate tensor
            gate = torch.rand(8, 320, 16, 8) * 0.6 + 0.2  # 模拟gate在[0.2, 0.8]范围
            monitor.update(gate, stage_name='stage2')
            
            gate3 = torch.rand(8, 640, 16, 8) * 0.5 + 0.25
            monitor.update(gate3, stage_name='stage3')
        
        monitor.end_epoch(epoch)
    
    monitor.summarize()
    monitor.plot_stats(save_path='./logs/gate_monitor/gate_stats.png')


