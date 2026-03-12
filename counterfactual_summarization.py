"""
Counterfactual Summarization Module

核心创新点：
1. 多策略反事实生成 (文本/视频)
2. 因果效应量化与损失计算
3. 可解释性分析接口

针对CNN/Daily Mail数据集设计，不依赖时序对齐。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import random


class CounterfactualGenerator:
    """
    生成反事实样本的工具类
    
    支持多种反事实生成策略：
    - zero: 将特征置零
    - noise: 添加高斯噪声
    - shuffle: 打乱序列顺序
    - mask: 随机掩码
    - mixed: 混合策略
    """
    
    STRATEGIES = ['zero', 'noise', 'shuffle', 'mask', 'mixed']
    
    def __init__(self, 
                 default_strategy: str = 'mixed',
                 noise_scale: float = 0.5,
                 mask_prob: float = 0.3):
        """
        Args:
            default_strategy: 默认使用的反事实生成策略
            noise_scale: 噪声强度 (用于noise策略)
            mask_prob: 掩码概率 (用于mask策略)
        """
        assert default_strategy in self.STRATEGIES, \
            f"Strategy must be one of {self.STRATEGIES}"
        self.default_strategy = default_strategy
        self.noise_scale = noise_scale
        self.mask_prob = mask_prob
    
    def generate_text_cf(self, 
                         text: torch.Tensor, 
                         strategy: Optional[str] = None,
                         intensity: float = 1.0,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成文本的反事实样本
        
        Args:
            text: 文本特征 [B, T, D]
            strategy: 使用的策略 (None则使用默认策略)
            intensity: 干预强度 (0-1)
            mask: 有效位置掩码 [B, T]
            
        Returns:
            反事实文本特征 [B, T, D]
        """
        strategy = strategy or self.default_strategy
        
        if strategy == 'mixed':
            strategy = random.choice(['zero', 'noise', 'shuffle', 'mask'])
        
        return self._apply_strategy(text, strategy, intensity, mask)
    
    def generate_video_cf(self, 
                          video: torch.Tensor, 
                          strategy: Optional[str] = None,
                          intensity: float = 1.0,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成视频的反事实样本
        
        Args:
            video: 视频特征 [B, T, D]
            strategy: 使用的策略
            intensity: 干预强度
            mask: 有效位置掩码 [B, T]
            
        Returns:
            反事实视频特征 [B, T, D]
        """
        strategy = strategy or self.default_strategy
        
        if strategy == 'mixed':
            strategy = random.choice(['zero', 'noise', 'shuffle', 'mask'])
        
        return self._apply_strategy(video, strategy, intensity, mask)
    
    def _apply_strategy(self, 
                        features: torch.Tensor, 
                        strategy: str,
                        intensity: float,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用具体的反事实生成策略
        """
        B, T, D = features.shape
        cf_features = features.clone()
        
        if strategy == 'zero':
            # 将部分特征置零
            num_to_zero = max(1, int(T * intensity))
            for b in range(B):
                if mask is not None:
                    valid_indices = torch.where(mask[b] > 0)[0]
                    if len(valid_indices) == 0:
                        continue
                    indices_to_zero = valid_indices[
                        torch.randperm(len(valid_indices))[:num_to_zero]
                    ]
                else:
                    indices_to_zero = torch.randperm(T)[:num_to_zero]
                cf_features[b, indices_to_zero] = 0
                
        elif strategy == 'noise':
            # 添加高斯噪声
            noise = torch.randn_like(features) * self.noise_scale * intensity
            if mask is not None:
                noise = noise * mask.unsqueeze(-1)
            cf_features = cf_features + noise
            
        elif strategy == 'shuffle':
            # 打乱序列顺序
            for b in range(B):
                if mask is not None:
                    valid_indices = torch.where(mask[b] > 0)[0]
                    if len(valid_indices) <= 1:
                        continue
                    # 部分打乱
                    num_to_shuffle = max(2, int(len(valid_indices) * intensity))
                    shuffle_indices = valid_indices[
                        torch.randperm(len(valid_indices))[:num_to_shuffle]
                    ]
                    shuffled = shuffle_indices[torch.randperm(len(shuffle_indices))]
                    cf_features[b, shuffle_indices] = features[b, shuffled]
                else:
                    perm = torch.randperm(T)
                    cf_features[b] = features[b, perm]
                    
        elif strategy == 'mask':
            # 随机掩码
            mask_tensor = torch.bernoulli(
                torch.ones(B, T, 1, device=features.device) * (1 - self.mask_prob * intensity)
            )
            if mask is not None:
                # 只掩码有效位置
                mask_tensor = mask_tensor * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1))
            cf_features = cf_features * mask_tensor
            
        return cf_features
    
    def generate_targeted_cf(self,
                             features: torch.Tensor,
                             target_indices: List[int],
                             strategy: str = 'zero',
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        针对特定位置生成反事实
        
        Args:
            features: 输入特征 [B, T, D]
            target_indices: 目标位置列表
            strategy: 使用的策略
            mask: 有效位置掩码
            
        Returns:
            反事实特征 [B, T, D]
        """
        cf_features = features.clone()
        target_tensor = torch.tensor(target_indices, device=features.device)
        
        if strategy == 'zero':
            cf_features[:, target_tensor] = 0
        elif strategy == 'noise':
            noise = torch.randn_like(cf_features[:, target_tensor]) * self.noise_scale
            cf_features[:, target_tensor] = cf_features[:, target_tensor] + noise
        elif strategy == 'mask':
            cf_features[:, target_tensor] = cf_features[:, target_tensor] * 0.1
            
        return cf_features


class CounterfactualSummarizationLoss(nn.Module):
    """
    反事实可解释摘要损失函数
    
    包含三个损失组件：
    1. 反事实对比损失 (cf_contrast_loss): 确保原始样本和反事实样本在特征空间中分离
    2. 因果效应损失 (cf_effect_loss): 确保反事实干预产生有意义的预测变化
    3. 敏感度损失 (cf_sensitivity_loss): 确保模型对关键信息敏感
    """
    
    def __init__(self,
                 lambda_cf_contrast: float = 0.5,
                 lambda_cf_effect: float = 0.3,
                 lambda_cf_sensitivity: float = 0.2,
                 enable_text_cf: bool = True,
                 enable_video_cf: bool = True,
                 cf_strategy: str = 'mixed',
                 temperature: float = 0.1):
        """
        Args:
            lambda_cf_contrast: 对比损失权重
            lambda_cf_effect: 因果效应损失权重
            lambda_cf_sensitivity: 敏感度损失权重
            enable_text_cf: 是否启用文本反事实
            enable_video_cf: 是否启用视频反事实
            cf_strategy: 反事实生成策略
            temperature: 对比学习温度参数
        """
        super().__init__()
        self.lambda_cf_contrast = lambda_cf_contrast
        self.lambda_cf_effect = lambda_cf_effect
        self.lambda_cf_sensitivity = lambda_cf_sensitivity
        self.enable_text_cf = enable_text_cf
        self.enable_video_cf = enable_video_cf
        self.temperature = temperature
        
        self.cf_generator = CounterfactualGenerator(
            default_strategy=cf_strategy
        )
        
    def forward(self,
                model: nn.Module,
                video: torch.Tensor,
                text: torch.Tensor,
                mask_video: torch.Tensor,
                mask_text: torch.Tensor,
                video_label: torch.Tensor,
                text_label: torch.Tensor,
                video_to_text_mask_list: List[torch.Tensor],
                text_to_video_mask_list: List[torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算反事实摘要损失
        
        Returns:
            包含各项损失的字典
        """
        device = video.device
        losses = {
            'cf_contrast_loss': torch.tensor(0.0, device=device),
            'cf_effect_loss': torch.tensor(0.0, device=device),
            'cf_sensitivity_loss': torch.tensor(0.0, device=device),
            'total_cf_loss': torch.tensor(0.0, device=device)
        }
        
        # 获取原始预测
        pred_video_orig, pred_text_orig, contrastive_pairs_orig = model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        # 计算文本反事实损失
        if self.enable_text_cf:
            text_cf = self.cf_generator.generate_text_cf(
                text, intensity=0.7, mask=mask_text
            )
            text_cf_losses = self._compute_cf_losses(
                model, video, text, text_cf,
                mask_video, mask_text,
                pred_video_orig, pred_text_orig,
                video_label, text_label,
                video_to_text_mask_list, text_to_video_mask_list,
                modality='text'
            )
            for k, v in text_cf_losses.items():
                losses[k] = losses[k] + v
        
        # 计算视频反事实损失
        if self.enable_video_cf:
            video_cf = self.cf_generator.generate_video_cf(
                video, intensity=0.7, mask=mask_video
            )
            video_cf_losses = self._compute_cf_losses(
                model, video_cf, text, None,
                mask_video, mask_text,
                pred_video_orig, pred_text_orig,
                video_label, text_label,
                video_to_text_mask_list, text_to_video_mask_list,
                modality='video'
            )
            for k, v in video_cf_losses.items():
                losses[k] = losses[k] + v
        
        # 计算总损失
        losses['total_cf_loss'] = (
            self.lambda_cf_contrast * losses['cf_contrast_loss'] +
            self.lambda_cf_effect * losses['cf_effect_loss'] +
            self.lambda_cf_sensitivity * losses['cf_sensitivity_loss']
        )
        
        return losses
    
    def _compute_cf_losses(self,
                           model: nn.Module,
                           video: torch.Tensor,
                           text_orig: torch.Tensor,
                           text_cf: Optional[torch.Tensor],
                           mask_video: torch.Tensor,
                           mask_text: torch.Tensor,
                           pred_video_orig: torch.Tensor,
                           pred_text_orig: torch.Tensor,
                           video_label: torch.Tensor,
                           text_label: torch.Tensor,
                           video_to_text_mask_list: List[torch.Tensor],
                           text_to_video_mask_list: List[torch.Tensor],
                           modality: str = 'text') -> Dict[str, torch.Tensor]:
        """
        计算单一模态的反事实损失
        """
        device = video.device
        
        # 选择反事实输入
        if modality == 'text':
            text_input = text_cf
            video_input = video
        else:  # video
            text_input = text_orig
            video_input = video  # video已经是反事实版本
        
        # 获取反事实预测
        pred_video_cf, pred_text_cf, contrastive_pairs_cf = model(
            video=video_input, text=text_input,
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        # 1. 对比损失: 确保原始和反事实表示分离
        cf_contrast_loss = self._contrastive_loss(
            contrastive_pairs_cf.get('video_fused', pred_video_cf),
            contrastive_pairs_cf.get('video_fused', pred_video_orig)
        )
        
        # 2. 因果效应损失: 反事实干预应产生预测变化
        cf_effect_loss = self._effect_loss(
            pred_video_orig, pred_video_cf,
            pred_text_orig, pred_text_cf,
            video_label, text_label,
            mask_video, mask_text
        )
        
        # 3. 敏感度损失: 对关键位置的干预应产生更大变化
        cf_sensitivity_loss = self._sensitivity_loss(
            pred_video_orig, pred_video_cf,
            pred_text_orig, pred_text_cf,
            video_label, text_label,
            mask_video, mask_text
        )
        
        return {
            'cf_contrast_loss': cf_contrast_loss,
            'cf_effect_loss': cf_effect_loss,
            'cf_sensitivity_loss': cf_sensitivity_loss
        }
    
    def _contrastive_loss(self, 
                          feat_orig: torch.Tensor, 
                          feat_cf: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失 - 确保原始和反事实表示在特征空间中分离
        """
        # 归一化
        feat_orig_norm = F.normalize(feat_orig.flatten(1), dim=-1)
        feat_cf_norm = F.normalize(feat_cf.flatten(1), dim=-1)
        
        # 相似度矩阵
        sim = torch.matmul(feat_orig_norm, feat_cf_norm.T) / self.temperature
        
        # 对角线应该较低 (同一样本的原始和反事实应该不同)
        # 这里我们希望对角线相似度低
        diag_sim = torch.diag(sim)
        
        # 损失: 希望对角线相似度低
        loss = torch.mean(F.softplus(diag_sim))
        
        return loss
    
    def _effect_loss(self,
                     pred_video_orig: torch.Tensor,
                     pred_video_cf: torch.Tensor,
                     pred_text_orig: torch.Tensor,
                     pred_text_cf: torch.Tensor,
                     video_label: torch.Tensor,
                     text_label: torch.Tensor,
                     mask_video: torch.Tensor,
                     mask_text: torch.Tensor) -> torch.Tensor:
        """
        计算因果效应损失 - 确保反事实干预能产生有意义的预测变化
        """
        # 视频预测变化
        video_diff = torch.abs(
            torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_cf)
        )
        video_diff = video_diff * mask_video
        
        # 文本预测变化
        text_diff = torch.abs(
            torch.sigmoid(pred_text_orig) - torch.sigmoid(pred_text_cf)
        )
        text_diff = text_diff * mask_text
        
        # 我们希望有效果 (diff > 0)，所以用负对数
        video_effect = -torch.log(video_diff.sum(dim=1) / mask_video.sum(dim=1) + 1e-8)
        text_effect = -torch.log(text_diff.sum(dim=1) / mask_text.sum(dim=1) + 1e-8)
        
        loss = video_effect.mean() + text_effect.mean()
        
        return loss
    
    def _sensitivity_loss(self,
                          pred_video_orig: torch.Tensor,
                          pred_video_cf: torch.Tensor,
                          pred_text_orig: torch.Tensor,
                          pred_text_cf: torch.Tensor,
                          video_label: torch.Tensor,
                          text_label: torch.Tensor,
                          mask_video: torch.Tensor,
                          mask_text: torch.Tensor) -> torch.Tensor:
        """
        计算敏感度损失 - 关键帧/句子应该对预测有更大影响
        """
        # 视频敏感度: 关键帧被干扰时应该产生更大变化
        video_diff = torch.abs(
            torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_cf)
        )
        
        # 关键帧位置 (label=1)
        key_frame_mask = video_label * mask_video
        non_key_mask = (1 - video_label) * mask_video
        
        # 关键帧的敏感度应该更高
        if key_frame_mask.sum() > 0 and non_key_mask.sum() > 0:
            key_sensitivity = (video_diff * key_frame_mask).sum() / (key_frame_mask.sum() + 1e-8)
            non_key_sensitivity = (video_diff * non_key_mask).sum() / (non_key_mask.sum() + 1e-8)
            # 希望 key > non_key
            video_sensitivity_loss = F.relu(non_key_sensitivity - key_sensitivity + 0.1)
        else:
            video_sensitivity_loss = torch.tensor(0.0, device=pred_video_orig.device)
        
        # 文本敏感度
        text_diff = torch.abs(
            torch.sigmoid(pred_text_orig) - torch.sigmoid(pred_text_cf)
        )
        
        key_sentence_mask = text_label * mask_text
        non_key_sentence_mask = (1 - text_label) * mask_text
        
        if key_sentence_mask.sum() > 0 and non_key_sentence_mask.sum() > 0:
            key_text_sensitivity = (text_diff * key_sentence_mask).sum() / (key_sentence_mask.sum() + 1e-8)
            non_key_text_sensitivity = (text_diff * non_key_sentence_mask).sum() / (non_key_sentence_mask.sum() + 1e-8)
            text_sensitivity_loss = F.relu(non_key_text_sensitivity - key_text_sensitivity + 0.1)
        else:
            text_sensitivity_loss = torch.tensor(0.0, device=pred_text_orig.device)
        
        return video_sensitivity_loss + text_sensitivity_loss


class ExplainableSummarizer:
    """
    可解释摘要分析器
    
    提供自然语言解释功能:
    - 为什么选择某个视频帧
    - 为什么选择某个句子
    - 如果删除某个元素会怎样 (反事实分析)
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 训练好的摘要模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.cf_generator = CounterfactualGenerator()
        self.model.eval()
    
    @torch.no_grad()
    def explain_frame_selection(self,
                                video: torch.Tensor,
                                text: torch.Tensor,
                                mask_video: torch.Tensor,
                                mask_text: torch.Tensor,
                                frame_idx: int,
                                **kwargs) -> Dict:
        """
        解释为什么选择某个视频帧
        
        Args:
            video: 视频特征 [1, T_v, D]
            text: 文本特征 [1, T_t, D]
            mask_video: 视频掩码
            mask_text: 文本掩码
            frame_idx: 要解释的帧索引
            
        Returns:
            解释字典
        """
        # 原始预测
        pred_video, pred_text, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        orig_score = torch.sigmoid(pred_video[0, frame_idx]).item()
        
        # 计算每个句子对该帧的贡献
        sentence_contributions = []
        num_sentences = int(mask_text.sum().item())
        
        for sent_idx in range(num_sentences):
            # 移除该句子
            text_cf = text.clone()
            text_cf[0, sent_idx] = 0
            
            pred_video_cf, _, _ = self.model(
                video=video, text=text_cf,
                mask_video=mask_video, mask_text=mask_text,
                **kwargs
            )
            cf_score = torch.sigmoid(pred_video_cf[0, frame_idx]).item()
            
            contribution = orig_score - cf_score
            sentence_contributions.append({
                'sentence_idx': sent_idx,
                'contribution': contribution,
                'direction': 'positive' if contribution > 0 else 'negative'
            })
        
        # 按贡献度排序
        sentence_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'frame_idx': frame_idx,
            'original_score': orig_score,
            'top_contributing_sentences': sentence_contributions[:5],
            'explanation': self._generate_frame_explanation(
                frame_idx, orig_score, sentence_contributions[:3]
            )
        }
    
    @torch.no_grad()
    def explain_sentence_selection(self,
                                   video: torch.Tensor,
                                   text: torch.Tensor,
                                   mask_video: torch.Tensor,
                                   mask_text: torch.Tensor,
                                   sentence_idx: int,
                                   **kwargs) -> Dict:
        """
        解释为什么选择某个句子
        """
        # 原始预测
        pred_video, pred_text, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        orig_score = torch.sigmoid(pred_text[0, sentence_idx]).item()
        
        # 计算每帧对该句子的贡献
        frame_contributions = []
        num_frames = int(mask_video.sum().item())
        
        for frame_idx in range(num_frames):
            # 移除该帧
            video_cf = video.clone()
            video_cf[0, frame_idx] = 0
            
            _, pred_text_cf, _ = self.model(
                video=video_cf, text=text,
                mask_video=mask_video, mask_text=mask_text,
                **kwargs
            )
            cf_score = torch.sigmoid(pred_text_cf[0, sentence_idx]).item()
            
            contribution = orig_score - cf_score
            frame_contributions.append({
                'frame_idx': frame_idx,
                'contribution': contribution,
                'direction': 'positive' if contribution > 0 else 'negative'
            })
        
        frame_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'sentence_idx': sentence_idx,
            'original_score': orig_score,
            'top_contributing_frames': frame_contributions[:5],
            'explanation': self._generate_sentence_explanation(
                sentence_idx, orig_score, frame_contributions[:3]
            )
        }
    
    @torch.no_grad()
    def what_if_remove_sentence(self,
                                video: torch.Tensor,
                                text: torch.Tensor,
                                mask_video: torch.Tensor,
                                mask_text: torch.Tensor,
                                sentence_idx: int,
                                **kwargs) -> Dict:
        """
        反事实分析: 如果删除某句话会怎样
        """
        # 原始预测
        pred_video_orig, pred_text_orig, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        # 删除句子后的预测
        text_cf = text.clone()
        text_cf[0, sentence_idx] = 0
        
        pred_video_cf, pred_text_cf, _ = self.model(
            video=video, text=text_cf,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        # 计算影响
        video_impact = (torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_cf))[0]
        text_impact = (torch.sigmoid(pred_text_orig) - torch.sigmoid(pred_text_cf))[0]
        
        # 找出受影响最大的帧
        num_frames = int(mask_video.sum().item())
        frame_impacts = [(i, video_impact[i].item()) for i in range(num_frames)]
        frame_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'removed_sentence_idx': sentence_idx,
            'total_video_impact': video_impact.abs().sum().item(),
            'total_text_impact': text_impact.abs().sum().item(),
            'most_affected_frames': frame_impacts[:5],
            'explanation': self._generate_removal_explanation(
                'sentence', sentence_idx, frame_impacts[:3]
            )
        }
    
    @torch.no_grad()
    def what_if_remove_frame(self,
                             video: torch.Tensor,
                             text: torch.Tensor,
                             mask_video: torch.Tensor,
                             mask_text: torch.Tensor,
                             frame_idx: int,
                             **kwargs) -> Dict:
        """
        反事实分析: 如果删除某帧会怎样
        """
        # 原始预测
        pred_video_orig, pred_text_orig, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        # 删除帧后的预测
        video_cf = video.clone()
        video_cf[0, frame_idx] = 0
        
        pred_video_cf, pred_text_cf, _ = self.model(
            video=video_cf, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        # 计算影响
        video_impact = (torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_cf))[0]
        text_impact = (torch.sigmoid(pred_text_orig) - torch.sigmoid(pred_text_cf))[0]
        
        # 找出受影响最大的句子
        num_sentences = int(mask_text.sum().item())
        sentence_impacts = [(i, text_impact[i].item()) for i in range(num_sentences)]
        sentence_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'removed_frame_idx': frame_idx,
            'total_video_impact': video_impact.abs().sum().item(),
            'total_text_impact': text_impact.abs().sum().item(),
            'most_affected_sentences': sentence_impacts[:5],
            'explanation': self._generate_removal_explanation(
                'frame', frame_idx, sentence_impacts[:3]
            )
        }
    
    def _generate_frame_explanation(self, 
                                    frame_idx: int, 
                                    score: float,
                                    top_sentences: List[Dict]) -> str:
        """生成帧选择解释的自然语言"""
        strength = "highly" if score > 0.7 else "moderately" if score > 0.4 else "marginally"
        
        explanation = f"Frame {frame_idx} is {strength} selected (score: {score:.2f}).\n"
        
        if top_sentences:
            explanation += "This selection is most influenced by:\n"
            for i, sent in enumerate(top_sentences, 1):
                direction = "supports" if sent['contribution'] > 0 else "opposes"
                explanation += f"  {i}. Sentence {sent['sentence_idx']} ({direction}, impact: {abs(sent['contribution']):.3f})\n"
        
        return explanation
    
    def _generate_sentence_explanation(self,
                                       sentence_idx: int,
                                       score: float,
                                       top_frames: List[Dict]) -> str:
        """生成句子选择解释的自然语言"""
        strength = "highly" if score > 0.7 else "moderately" if score > 0.4 else "marginally"
        
        explanation = f"Sentence {sentence_idx} is {strength} selected (score: {score:.2f}).\n"
        
        if top_frames:
            explanation += "This selection is most influenced by:\n"
            for i, frame in enumerate(top_frames, 1):
                direction = "supports" if frame['contribution'] > 0 else "opposes"
                explanation += f"  {i}. Frame {frame['frame_idx']} ({direction}, impact: {abs(frame['contribution']):.3f})\n"
        
        return explanation
    
    def _generate_removal_explanation(self,
                                      element_type: str,
                                      element_idx: int,
                                      impacts: List[Tuple[int, float]]) -> str:
        """生成移除元素后影响的解释"""
        other_type = "frame" if element_type == "sentence" else "sentence"
        
        explanation = f"If {element_type} {element_idx} were removed:\n"
        
        total_impact = sum(abs(imp[1]) for imp in impacts)
        if total_impact > 0.5:
            explanation += "  → Significant changes would occur:\n"
        elif total_impact > 0.2:
            explanation += "  → Moderate changes would occur:\n"
        else:
            explanation += "  → Minimal changes would occur:\n"
        
        for idx, impact in impacts:
            direction = "decrease" if impact > 0 else "increase"
            explanation += f"    • {other_type.capitalize()} {idx} would {direction} by {abs(impact):.3f}\n"
        
        return explanation


if __name__ == '__main__':
    print("Testing CounterfactualSummarization module...")
    
    # Test CounterfactualGenerator
    print("\n1. Testing CounterfactualGenerator...")
    cf_gen = CounterfactualGenerator(default_strategy='mixed')
    
    dummy_text = torch.randn(2, 10, 768)
    dummy_mask = torch.ones(2, 10)
    dummy_mask[0, 7:] = 0
    dummy_mask[1, 8:] = 0
    
    for strategy in ['zero', 'noise', 'shuffle', 'mask']:
        cf_text = cf_gen.generate_text_cf(dummy_text, strategy=strategy, mask=dummy_mask)
        print(f"  ✓ Strategy '{strategy}': output shape = {cf_text.shape}")
    
    # Test targeted CF
    target_cf = cf_gen.generate_targeted_cf(dummy_text, [0, 2, 4], strategy='zero')
    print(f"  ✓ Targeted CF: modified positions [0, 2, 4]")
    
    print("\n2. Testing CounterfactualSummarizationLoss...")
    cf_loss_fn = CounterfactualSummarizationLoss(
        lambda_cf_contrast=0.5,
        lambda_cf_effect=0.3,
        lambda_cf_sensitivity=0.2,
        enable_text_cf=True,
        enable_video_cf=True
    )
    print(f"  ✓ CounterfactualSummarizationLoss initialized")
    print(f"    - lambda_cf_contrast: {cf_loss_fn.lambda_cf_contrast}")
    print(f"    - lambda_cf_effect: {cf_loss_fn.lambda_cf_effect}")
    print(f"    - lambda_cf_sensitivity: {cf_loss_fn.lambda_cf_sensitivity}")
    
    print("\n✓ All module tests passed!")
    print("\nNote: Full forward pass requires a trained model.")
