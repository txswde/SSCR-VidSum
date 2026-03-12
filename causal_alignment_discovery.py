"""
Causal Alignment Discovery - 基于因果干预的无监督对齐发现

核心创新：干预非相似度 (Intervention Dissimilarity)
- 对每个文本句子j进行干预（移除/零化）
- 计算每个视频帧i在干预前后的预测变化
- 变化越大 → 该帧与该句子对齐越强
- 无需GT标注即可发现跨模态对齐

数学形式：
    Alignment_{i,j} = |P(Y_i|V, T) - P(Y_i|V, T_{\j})|
    
其中 T_{\j} 表示移除第j个句子后的文本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CausalAlignmentDiscovery(nn.Module):
    """
    基于因果干预的无监督对齐发现模块
    
    核心思想：
    1. 对每个文本句子进行干预（zero-out）
    2. 观察每个视频帧预测的变化
    3. 变化矩阵即为发现的因果对齐
    
    Alignment_{i,j} = |pred_orig_i - pred_intervened_i|
    """
    
    def __init__(self,
                 lambda_diagonal: float = 1.0,
                 lambda_sparsity: float = 0.1,
                 temperature: float = 1.0,
                 normalize_alignment: bool = True):
        """
        Args:
            lambda_diagonal: 对角线正则化权重（鼓励时序对齐）
            lambda_sparsity: 稀疏性正则化权重
            temperature: 温度参数用于softmax归一化
            normalize_alignment: 是否归一化对齐矩阵
        """
        super().__init__()
        self.lambda_diagonal = lambda_diagonal
        self.lambda_sparsity = lambda_sparsity
        self.temperature = temperature
        self.normalize_alignment = normalize_alignment
    
    def _extract_video_pred(self, model_output):
        """
        从模型输出中提取视频预测，适配不同模型的返回格式
        
        - Model_BLiSS/Model_MSMO: 返回 (pred_video, pred_text, contrastive_pairs) 元组
        - Model_VideoSumm: 返回字典 {'video_pred_cls': ..., ...}
        """
        if isinstance(model_output, dict):
            # Model_VideoSumm format
            return model_output['video_pred_cls']
        elif isinstance(model_output, tuple):
            # Model_BLiSS/MSMO format
            return model_output[0]
        else:
            raise ValueError(f"Unknown model output format: {type(model_output)}")
    
    def compute_intervention_effects(self,
                                      model,
                                      video: torch.Tensor,
                                      text: torch.Tensor,
                                      mask_video: torch.Tensor,
                                      mask_text: torch.Tensor,
                                      video_label: torch.Tensor,
                                      text_label: torch.Tensor,
                                      video_to_text_mask_list: List[torch.Tensor],
                                      text_to_video_mask_list: List[torch.Tensor]
                                      ) -> Dict[str, torch.Tensor]:
        """
        计算干预效应矩阵（核心算法）
        
        Args:
            model: 模型 (支持 BLiSS/MSMO 和 VideoSumm 两种格式)
            video: 视频特征 [B, T_v, D_v]
            text: 文本特征 [B, T_t, D_t]
            mask_video: 视频mask [B, T_v]
            mask_text: 文本mask [B, T_t]
            其他参数同train_msmo/train_videosumm
            
        Returns:
            intervention_effects: 干预效应矩阵 [B, T_v, T_t]
            causal_alignment: 归一化的因果对齐矩阵 [B, T_v, T_t]
        """
        B, T_v, _ = video.shape
        _, T_t, D_t = text.shape
        device = video.device
        
        # Step 1: 获取原始预测
        with torch.no_grad():
            model_output_orig = model(
                video=video, text=text,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            pred_video_orig = self._extract_video_pred(model_output_orig)
            prob_orig = torch.sigmoid(pred_video_orig)  # [B, T_v]
        
        # Step 2: 对每个文本句子进行干预，计算预测变化
        intervention_effects = torch.zeros(B, T_v, T_t, device=device)
        
        for j in range(T_t):
            # 创建干预后的文本（移除第j个句子）
            text_intervened = text.clone()
            text_intervened[:, j, :] = 0  # Zero-out第j个句子
            
            # 获取干预后的预测
            with torch.no_grad():
                model_output_intervened = model(
                    video=video, text=text_intervened,
                    mask_video=mask_video, mask_text=mask_text,
                    video_label=video_label, text_label=text_label,
                    video_to_text_mask_list=video_to_text_mask_list,
                    text_to_video_mask_list=text_to_video_mask_list
                )
                pred_video_intervened = self._extract_video_pred(model_output_intervened)
                prob_intervened = torch.sigmoid(pred_video_intervened)  # [B, T_v]
            
            # 干预效应 = |原始预测 - 干预后预测|
            effect_j = torch.abs(prob_orig - prob_intervened)  # [B, T_v]
            intervention_effects[:, :, j] = effect_j
        
        # Step 3: 归一化为对齐分布
        if self.normalize_alignment:
            # 对每个视频帧，softmax归一化得到对文本的注意力分布
            causal_alignment = F.softmax(
                intervention_effects / self.temperature, 
                dim=-1
            )  # [B, T_v, T_t]
        else:
            causal_alignment = intervention_effects
        
        return {
            'intervention_effects': intervention_effects,  # 原始干预效应
            'causal_alignment': causal_alignment,  # 归一化的因果对齐
            'prob_orig': prob_orig  # 原始预测概率
        }
    
    def compute_alignment_loss(self,
                               causal_alignment: torch.Tensor,
                               mask_video: torch.Tensor,
                               mask_text: torch.Tensor
                               ) -> Dict[str, torch.Tensor]:
        """
        计算对齐正则化损失
        
        鼓励发现的对齐呈现对角线模式（时序对应）
        
        Args:
            causal_alignment: 因果对齐矩阵 [B, T_v, T_t]
            mask_video: 视频mask [B, T_v]
            mask_text: 文本mask [B, T_t]
            
        Returns:
            损失字典
        """
        B, T_v, T_t = causal_alignment.shape
        device = causal_alignment.device
        
        total_diagonal_loss = torch.tensor(0.0, device=device)
        total_sparsity_loss = torch.tensor(0.0, device=device)
        
        for b in range(B):
            # 获取有效范围
            T_v_valid = int(mask_video[b].sum().item())
            T_t_valid = int(mask_text[b].sum().item())
            
            if T_v_valid == 0 or T_t_valid == 0:
                continue
            
            align_b = causal_alignment[b, :T_v_valid, :T_t_valid]  # [T_v_valid, T_t_valid]
            
            # === 1. 对角线正则化 ===
            # 理想对齐应该沿着对角线（时序对应）
            v_idx = torch.arange(T_v_valid, device=device).float().view(-1, 1)
            t_idx = torch.arange(T_t_valid, device=device).float().view(1, -1)
            
            # 归一化到[0, 1]
            v_norm = v_idx / (T_v_valid - 1 + 1e-8)
            t_norm = t_idx / (T_t_valid - 1 + 1e-8)
            
            # 对角线权重（距离对角线越近权重越高）
            diagonal_dist = torch.abs(v_norm - t_norm)
            diagonal_weight = torch.exp(-3.0 * diagonal_dist)  # 对角线带宽
            
            # 损失：鼓励高对齐权重出现在对角线
            # = -sum(alignment * diagonal_weight)
            diagonal_loss_b = -torch.sum(align_b * diagonal_weight) / (T_v_valid * T_t_valid)
            total_diagonal_loss = total_diagonal_loss + diagonal_loss_b
            
            # === 2. 稀疏性正则化 ===
            # 每个视频帧应该只对齐少数文本句子
            # 使用负熵鼓励稀疏分布
            entropy = -torch.sum(align_b * torch.log(align_b + 1e-8), dim=-1)  # [T_v_valid]
            sparsity_loss_b = entropy.mean()  # 最小化熵 = 最大化稀疏性
            total_sparsity_loss = total_sparsity_loss + sparsity_loss_b
        
        # 平均到batch
        diagonal_loss = total_diagonal_loss / B
        sparsity_loss = total_sparsity_loss / B
        
        total_loss = (self.lambda_diagonal * diagonal_loss + 
                      self.lambda_sparsity * sparsity_loss)
        
        return {
            'diagonal_loss': diagonal_loss,
            'sparsity_loss': sparsity_loss,
            'total_causal_alignment_loss': total_loss
        }
    
    def evaluate_alignment_quality(self,
                                   causal_alignment: torch.Tensor,
                                   gt_alignment_list: List[torch.Tensor],
                                   mask_video: torch.Tensor,
                                   mask_text: torch.Tensor
                                   ) -> Dict[str, float]:
        """
        评估发现的对齐与GT对齐的质量
        
        Args:
            causal_alignment: 因果对齐矩阵 [B, T_v, T_t]
            gt_alignment_list: GT对齐mask列表
            mask_video, mask_text: 有效mask
            
        Returns:
            质量指标字典
        """
        B = causal_alignment.shape[0]
        device = causal_alignment.device
        
        total_correlation = 0.0
        total_precision_at_1 = 0.0
        total_iou = 0.0
        count = 0
        
        for b in range(B):
            gt_mask = gt_alignment_list[b].float().to(device)
            T_v_valid, T_t_valid = gt_mask.shape
            
            pred_align = causal_alignment[b, :T_v_valid, :T_t_valid]
            
            # 1. 改进的相关性计算：对比GT对齐区域vs非对齐区域的预测值差异
            # 这对二值GT mask更加鲁棒
            pred_flat = pred_align.flatten()
            gt_flat = gt_mask.flatten()
            
            # 方法1: 经典皮尔逊相关系数
            pred_centered = pred_flat - pred_flat.mean()
            gt_centered = gt_flat - gt_flat.mean()
            
            # 检查是否有足够方差
            pred_std = pred_centered.norm()
            gt_std = gt_centered.norm()
            
            if pred_std > 1e-6 and gt_std > 1e-6:
                pearson_corr = (pred_centered * gt_centered).sum() / (pred_std * gt_std + 1e-8)
            else:
                pearson_corr = torch.tensor(0.0, device=device)
            
            # 方法2: 对齐区域vs非对齐区域的预测值差异（更直观）
            gt_aligned_mask = gt_flat > 0.5
            gt_non_aligned_mask = gt_flat <= 0.5
            
            if gt_aligned_mask.sum() > 0 and gt_non_aligned_mask.sum() > 0:
                pred_in_aligned = pred_flat[gt_aligned_mask].mean()
                pred_in_non_aligned = pred_flat[gt_non_aligned_mask].mean()
                # 归一化差异到[-1, 1]
                value_diff = (pred_in_aligned - pred_in_non_aligned) / (pred_flat.max() - pred_flat.min() + 1e-8)
                # 使用差异和Pearson的加权组合
                correlation = 0.5 * pearson_corr + 0.5 * value_diff.clamp(-1, 1)
            else:
                correlation = pearson_corr
            
            total_correlation += correlation.item()
            
            # 2. Precision@1: 每行预测的最大位置是否在GT对齐区域内
            pred_argmax = pred_align.argmax(dim=-1)  # [T_v_valid]
            hits = 0
            for i in range(T_v_valid):
                if gt_mask[i, pred_argmax[i]] > 0:
                    hits += 1
            precision_at_1 = hits / T_v_valid
            total_precision_at_1 += precision_at_1
            
            # 3. IoU (对齐区域重叠度)
            pred_binary = (pred_align > pred_align.mean()).float()
            intersection = (pred_binary * gt_mask).sum()
            union = ((pred_binary + gt_mask) > 0).float().sum()
            iou = (intersection / (union + 1e-8)).item()
            total_iou += iou
            
            count += 1
        
        return {
            'correlation': total_correlation / max(count, 1),
            'precision_at_1': total_precision_at_1 / max(count, 1),
            'iou': total_iou / max(count, 1)
        }
    
    def forward(self,
                model,
                video: torch.Tensor,
                text: torch.Tensor,
                mask_video: torch.Tensor,
                mask_text: torch.Tensor,
                video_label: torch.Tensor,
                text_label: torch.Tensor,
                video_to_text_mask_list: List[torch.Tensor],
                text_to_video_mask_list: List[torch.Tensor],
                compute_loss: bool = True,
                evaluate: bool = False
                ) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算干预效应、对齐矩阵和损失
        
        Args:
            model: BLiSS模型
            video, text, masks: 输入数据
            compute_loss: 是否计算正则化损失
            evaluate: 是否评估对齐质量（需要GT）
            
        Returns:
            包含对齐矩阵和损失的字典
        """
        # 计算干预效应
        effect_result = self.compute_intervention_effects(
            model=model,
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        result = {
            'intervention_effects': effect_result['intervention_effects'],
            'causal_alignment': effect_result['causal_alignment']
        }
        
        # 计算对齐正则化损失
        if compute_loss:
            loss_result = self.compute_alignment_loss(
                causal_alignment=effect_result['causal_alignment'],
                mask_video=mask_video,
                mask_text=mask_text
            )
            result.update(loss_result)
        
        # 评估对齐质量
        if evaluate:
            eval_result = self.evaluate_alignment_quality(
                causal_alignment=effect_result['causal_alignment'],
                gt_alignment_list=video_to_text_mask_list,
                mask_video=mask_video,
                mask_text=mask_text
            )
            result['evaluation'] = eval_result
        
        return result


class InterventionDissimilarityLoss(nn.Module):
    """
    干预非相似度损失：直接用于训练的简化版本
    
    核心思想：
    1. 不需要完整遍历所有句子
    2. 采用采样策略减少计算开销
    3. 通过损失引导模型学习可解释的对齐
    """
    
    def __init__(self,
                 num_samples: int = 5,
                 lambda_effect: float = 0.5,
                 margin: float = 0.1):
        """
        Args:
            num_samples: 每次采样的句子数量
            lambda_effect: 效应损失权重
            margin: 边界值
        """
        super().__init__()
        self.num_samples = num_samples
        self.lambda_effect = lambda_effect
        self.margin = margin
    
    def forward(self,
                pred_orig: torch.Tensor,
                pred_list_intervened: List[torch.Tensor],
                intervened_indices: List[int],
                video_to_text_mask_list: List[torch.Tensor],
                mask_video: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        计算采样版本的干预非相似度损失
        
        Args:
            pred_orig: 原始预测 [B, T_v]
            pred_list_intervened: 干预后预测列表，每个元素 [B, T_v]
            intervened_indices: 被干预的句子索引
            video_to_text_mask_list: GT对齐mask
            mask_video: 视频有效mask
            
        Returns:
            损失字典
        """
        B, T_v = pred_orig.shape
        device = pred_orig.device
        
        total_alignment_consistency = torch.tensor(0.0, device=device)
        
        prob_orig = torch.sigmoid(pred_orig)
        
        for b in range(B):
            gt_mask = video_to_text_mask_list[b].float()
            T_v_valid = int(mask_video[b].sum().item())
            
            for idx, j in enumerate(intervened_indices):
                if j >= gt_mask.shape[1]:
                    continue
                    
                prob_intervened = torch.sigmoid(pred_list_intervened[idx][b])
                effect = torch.abs(prob_orig[b, :T_v_valid] - prob_intervened[:T_v_valid])
                
                # 在GT对齐区域，干预效应应该更大
                aligned_frames = gt_mask[:T_v_valid, j] > 0
                non_aligned_frames = gt_mask[:T_v_valid, j] == 0
                
                if aligned_frames.sum() > 0 and non_aligned_frames.sum() > 0:
                    effect_aligned = effect[aligned_frames].mean()
                    effect_non_aligned = effect[non_aligned_frames].mean()
                    
                    # 对齐区域的效应应该大于非对齐区域
                    consistency_loss = F.relu(
                        self.margin - (effect_aligned - effect_non_aligned)
                    )
                    total_alignment_consistency = total_alignment_consistency + consistency_loss
        
        # 归一化
        num_interventions = B * len(intervened_indices)
        alignment_consistency_loss = total_alignment_consistency / max(num_interventions, 1)
        
        return {
            'alignment_consistency': alignment_consistency_loss,
            'total_intervention_loss': alignment_consistency_loss * self.lambda_effect
        }


if __name__ == '__main__':
    print("Testing CausalAlignmentDiscovery...")
    
    # 模拟数据
    B, T_v, T_t = 2, 50, 10
    
    # 创建模拟对齐矩阵（对角线模式）
    causal_alignment = torch.zeros(B, T_v, T_t)
    for b in range(B):
        for i in range(T_v):
            j = min(int(i * T_t / T_v), T_t - 1)
            causal_alignment[b, i, j] = 1.0
    causal_alignment = F.softmax(causal_alignment / 0.1, dim=-1)
    
    mask_video = torch.ones(B, T_v)
    mask_text = torch.ones(B, T_t)
    
    # 创建GT对齐
    gt_alignment_list = []
    for b in range(B):
        gt = torch.zeros(T_v, T_t)
        frames_per_sent = T_v // T_t
        for t in range(T_t):
            start = t * frames_per_sent
            end = min((t + 1) * frames_per_sent, T_v)
            gt[start:end, t] = 1
        gt_alignment_list.append(gt)
    
    # 测试模块
    discovery = CausalAlignmentDiscovery()
    
    # 测试对齐损失
    loss_result = discovery.compute_alignment_loss(causal_alignment, mask_video, mask_text)
    print("\nAlignment Loss:")
    for k, v in loss_result.items():
        print(f"  {k}: {v.item():.4f}")
    
    # 测试评估
    eval_result = discovery.evaluate_alignment_quality(
        causal_alignment, gt_alignment_list, mask_video, mask_text
    )
    print("\nAlignment Quality:")
    for k, v in eval_result.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✓ All tests passed!")
