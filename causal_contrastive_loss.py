"""
Causal Contrastive Loss Functions (Simplified)
核心创新: 因果对比学习

精简后仅保留两个核心组件:
1. CausalContrastiveLoss - 因果对比损失 (创新1)
2. SimplifiedCausalLoss - 用于 TVSum/SumMe 的简化版损失

干预相异度对齐 (创新2) 由 causal_alignment_discovery.py 独立处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CausalContrastiveLoss(nn.Module):
    """
    因果对比损失 (创新1核心模块)
    
    核心思想:
    1. 轻微干预 (minimal CF) → 特征应保持相似 (拉近)
    2. 剧烈干预 (severe CF) → 特征应产生变化 (推远)
    
    通过这种方式区分 "相关性" 和 "因果性"
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 margin: float = 0.5,
                 lambda_invariance: float = 0.1):
        """
        Args:
            temperature: 温度参数 (用于 softmax 归一化)
            margin: 边界值 (severe CF 应至少远离 margin)
            lambda_invariance: 不变性正则化权重
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.lambda_invariance = lambda_invariance
    
    def forward(self, 
                feat_orig: torch.Tensor,
                feat_cf_minimal: torch.Tensor,
                feat_cf_severe: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算因果对比损失
        
        Args:
            feat_orig: 原始特征 [B, T, D]
            feat_cf_minimal: 轻微干预特征 [B, T, D]
            feat_cf_severe: 剧烈干预特征 [B, T, D]
            
        Returns:
            损失字典
        """
        B, T, D = feat_orig.shape
        
        # Flatten for computation
        feat_orig_flat = feat_orig.reshape(-1, D)
        feat_cf_min_flat = feat_cf_minimal.reshape(-1, D)
        feat_cf_sev_flat = feat_cf_severe.reshape(-1, D)
        
        # 1. 拉近 minimal CF (应与原始相似)
        dist_minimal = F.mse_loss(feat_orig_flat, feat_cf_min_flat, reduction='mean')
        
        # 2. 推远 severe CF (应与原始不同)
        dist_severe = F.mse_loss(feat_orig_flat, feat_cf_sev_flat, reduction='mean')
        
        # Margin-based loss: minimize dist_minimal, maximize dist_severe
        contrast_loss = dist_minimal + F.relu(self.margin - dist_severe)
        
        # 3. 不变性正则化 (保持特征范数稳定)
        norm_orig = feat_orig.norm(dim=-1)
        norm_cf = feat_cf_minimal.norm(dim=-1)
        invariance_loss = F.mse_loss(norm_orig, norm_cf, reduction='mean')
        
        # Total
        total_loss = contrast_loss + self.lambda_invariance * invariance_loss
        
        return {
            'total': total_loss,
            'contrast': contrast_loss,
            'invariance': invariance_loss,
            'dist_minimal': dist_minimal,
            'dist_severe': dist_severe
        }


class SimplifiedCausalLoss(nn.Module):
    """
    简化版因果损失 (用于 TVSum/SumMe)
    
    仅包含一个核心参数: lambda_causal_contrast
    
    注意: 干预相异度对齐 (创新2) 由 CausalAlignmentDiscovery 独立处理,
         不在此类中实现，避免功能重叠。
    """
    
    def __init__(self,
                 lambda_causal_contrast: float = 0.3,
                 temperature: float = 0.07,
                 margin: float = 0.5):
        """
        Args:
            lambda_causal_contrast: 因果对比损失权重 (创新1)
        """
        super().__init__()
        self.lambda_causal_contrast = lambda_causal_contrast
        self.causal_contrast_loss = CausalContrastiveLoss(
            temperature=temperature,
            margin=margin
        )
    
    def forward(self, outputs: Dict, labels: Dict, masks: Dict) -> Dict[str, torch.Tensor]:
        """
        计算简化版因果损失
        
        Args:
            outputs: 模型输出，需包含:
                - text_feat_orig: 原始文本特征
                - text_feat_cf_min: 轻微干预文本特征
                - text_feat_cf_sev: 剧烈干预文本特征
            labels: 标签 (当前未使用)
            masks: 掩码 (当前未使用)
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 因果对比损失 (创新1)
        if (outputs.get('text_feat_orig') is not None and 
            outputs.get('text_feat_cf_min') is not None and 
            outputs.get('text_feat_cf_sev') is not None):
            
            contrast_dict = self.causal_contrast_loss(
                outputs['text_feat_orig'],
                outputs['text_feat_cf_min'],
                outputs['text_feat_cf_sev']
            )
            
            # 应用权重
            for k, v in contrast_dict.items():
                losses[f'causal_{k}'] = v * self.lambda_causal_contrast
        else:
            # 如果没有反事实特征，返回零损失
            device = next(iter(outputs.values())).device if outputs else 'cpu'
            losses['causal_total'] = torch.tensor(0.0, device=device)
        
        # 总损失
        losses['total_causal'] = sum(
            v for k, v in losses.items() 
            if k.startswith('causal_') and 'total' in k
        )
        
        return losses


# === 向后兼容别名 ===
# 保持 CombinedCausalLoss 名称以兼容现有代码
CombinedCausalLoss = SimplifiedCausalLoss


if __name__ == '__main__':
    print("Testing Simplified CausalContrastiveLoss...")
    
    B, T, D = 4, 10, 512
    
    # Create dummy features
    feat_orig = torch.randn(B, T, D)
    feat_cf_minimal = feat_orig + torch.randn(B, T, D) * 0.1  # Small perturbation
    feat_cf_severe = torch.randn(B, T, D)  # Large perturbation
    
    # Test CausalContrastiveLoss
    criterion = CausalContrastiveLoss()
    losses = criterion(feat_orig, feat_cf_minimal, feat_cf_severe)
    
    print("\nCausalContrastiveLoss 结果:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    # Test SimplifiedCausalLoss
    print("\n\nTesting SimplifiedCausalLoss...")
    outputs = {
        'text_feat_orig': feat_orig,
        'text_feat_cf_min': feat_cf_minimal,
        'text_feat_cf_sev': feat_cf_severe
    }
    
    simplified = SimplifiedCausalLoss(lambda_causal_contrast=0.3)
    losses2 = simplified(outputs, {}, {})
    
    print("\nSimplifiedCausalLoss 结果:")
    for name, value in losses2.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\n✓ 所有测试通过!")
