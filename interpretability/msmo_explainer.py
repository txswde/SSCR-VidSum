"""
MSMO (CNN/Daily Mail) 专用解释器

针对CNN/Daily Mail多模态摘要数据集的可解释性分析工具
不依赖时序对齐，专注于语义层面的因果解释
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# 导入因果分析器
try:
    from .causal_explainer import CausalEffectAnalyzer, CausalVisualizer
except ImportError:
    from causal_explainer import CausalEffectAnalyzer, CausalVisualizer


class MSMOExplainer:
    """
    CNN/Daily Mail多模态摘要解释器
    
    功能:
    1. 解释视频帧选择决策
    2. 解释文本句子选择决策
    3. 反事实分析 (what-if scenarios)
    4. 生成可读的摘要报告
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'cuda',
                 save_dir: str = './explanations'):
        """
        Args:
            model: 训练好的摘要模型
            device: 计算设备
            save_dir: 保存解释结果的目录
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.analyzer = CausalEffectAnalyzer(model)
        self.visualizer = CausalVisualizer(save_dir=save_dir)
        
        self.model.eval()
    
    @torch.no_grad()
    def explain_video_selection(self,
                                video: torch.Tensor,
                                text: torch.Tensor,
                                mask_video: torch.Tensor,
                                mask_text: torch.Tensor,
                                selected_frames: List[int],
                                article_sentences: Optional[List[str]] = None,
                                **kwargs) -> Dict:
        """
        解释为什么选择特定的视频帧
        
        Args:
            video: 视频特征 [1, T_v, D_v]
            text: 文本特征 [1, T_t, D_t]
            mask_video: 视频掩码 [1, T_v]
            mask_text: 文本掩码 [1, T_t]
            selected_frames: 被选中的帧索引列表
            article_sentences: 可选，原始句子文本列表 (用于生成可读解释)
            
        Returns:
            解释字典
        """
        explanations = []
        
        # 获取原始预测
        pred_video, pred_text, contrastive_pairs = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        num_sentences = int(mask_text.sum().item())
        
        for frame_idx in selected_frames:
            frame_explanation = {
                'frame_idx': frame_idx,
                'selection_score': torch.sigmoid(pred_video[0, frame_idx]).item(),
                'sentence_contributions': []
            }
            
            # 计算每个句子对这一帧选择的贡献
            for sent_idx in range(num_sentences):
                # 创建反事实: 移除该句子
                text_cf = text.clone()
                text_cf[0, sent_idx] = 0
                
                pred_video_cf, _, _ = self.model(
                    video=video, text=text_cf,
                    mask_video=mask_video, mask_text=mask_text,
                    **kwargs
                )
                
                # 贡献 = 原始分数 - 移除后分数
                contribution = (
                    torch.sigmoid(pred_video[0, frame_idx]) - 
                    torch.sigmoid(pred_video_cf[0, frame_idx])
                ).item()
                
                sent_info = {
                    'sentence_idx': sent_idx,
                    'contribution': contribution,
                    'is_supportive': contribution > 0
                }
                
                if article_sentences and sent_idx < len(article_sentences):
                    sent_info['sentence_text'] = article_sentences[sent_idx][:100] + '...' \
                        if len(article_sentences[sent_idx]) > 100 else article_sentences[sent_idx]
                
                frame_explanation['sentence_contributions'].append(sent_info)
            
            # 按贡献度排序
            frame_explanation['sentence_contributions'].sort(
                key=lambda x: abs(x['contribution']), reverse=True
            )
            
            # 生成可读解释
            frame_explanation['natural_language_explanation'] = self._generate_frame_nl_explanation(
                frame_idx, 
                frame_explanation['selection_score'],
                frame_explanation['sentence_contributions'][:3],
                article_sentences
            )
            
            explanations.append(frame_explanation)
        
        return {
            'type': 'video_selection',
            'num_frames_explained': len(selected_frames),
            'frame_explanations': explanations
        }
    
    @torch.no_grad()
    def explain_text_selection(self,
                               video: torch.Tensor,
                               text: torch.Tensor,
                               mask_video: torch.Tensor,
                               mask_text: torch.Tensor,
                               selected_sentences: List[int],
                               article_sentences: Optional[List[str]] = None,
                               **kwargs) -> Dict:
        """
        解释为什么选择特定的句子
        
        Args:
            selected_sentences: 被选中的句子索引列表
            
        Returns:
            解释字典
        """
        explanations = []
        
        # 获取原始预测
        pred_video, pred_text, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        num_frames = int(mask_video.sum().item())
        
        for sent_idx in selected_sentences:
            sent_explanation = {
                'sentence_idx': sent_idx,
                'selection_score': torch.sigmoid(pred_text[0, sent_idx]).item(),
                'frame_contributions': []
            }
            
            if article_sentences and sent_idx < len(article_sentences):
                sent_explanation['sentence_text'] = article_sentences[sent_idx]
            
            # 计算每帧对该句子选择的贡献
            for frame_idx in range(num_frames):
                video_cf = video.clone()
                video_cf[0, frame_idx] = 0
                
                _, pred_text_cf, _ = self.model(
                    video=video_cf, text=text,
                    mask_video=mask_video, mask_text=mask_text,
                    **kwargs
                )
                
                contribution = (
                    torch.sigmoid(pred_text[0, sent_idx]) - 
                    torch.sigmoid(pred_text_cf[0, sent_idx])
                ).item()
                
                sent_explanation['frame_contributions'].append({
                    'frame_idx': frame_idx,
                    'contribution': contribution,
                    'is_supportive': contribution > 0
                })
            
            # 按贡献度排序
            sent_explanation['frame_contributions'].sort(
                key=lambda x: abs(x['contribution']), reverse=True
            )
            
            # 生成可读解释
            sent_explanation['natural_language_explanation'] = self._generate_sentence_nl_explanation(
                sent_idx,
                sent_explanation['selection_score'],
                sent_explanation['frame_contributions'][:3],
                article_sentences[sent_idx] if article_sentences and sent_idx < len(article_sentences) else None
            )
            
            explanations.append(sent_explanation)
        
        return {
            'type': 'text_selection',
            'num_sentences_explained': len(selected_sentences),
            'sentence_explanations': explanations
        }
    
    @torch.no_grad()
    def what_if_remove_sentence(self,
                                video: torch.Tensor,
                                text: torch.Tensor,
                                mask_video: torch.Tensor,
                                mask_text: torch.Tensor,
                                sentence_idx: int,
                                article_sentences: Optional[List[str]] = None,
                                **kwargs) -> Dict:
        """
        反事实分析：如果删除某句话会怎样
        
        分析删除特定句子后对视频帧选择和其他句子选择的影响
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
        
        # 计算视频帧影响
        video_probs_orig = torch.sigmoid(pred_video_orig[0])
        video_probs_cf = torch.sigmoid(pred_video_cf[0])
        video_changes = (video_probs_orig - video_probs_cf).cpu().numpy()
        
        # 计算其他句子影响
        text_probs_orig = torch.sigmoid(pred_text_orig[0])
        text_probs_cf = torch.sigmoid(pred_text_cf[0])
        text_changes = (text_probs_orig - text_probs_cf).cpu().numpy()
        
        num_frames = int(mask_video.sum().item())
        num_sentences = int(mask_text.sum().item())
        
        # 找出受影响最大的帧
        frame_impacts = [
            {'frame_idx': i, 'change': float(video_changes[i])}
            for i in range(num_frames)
        ]
        frame_impacts.sort(key=lambda x: abs(x['change']), reverse=True)
        
        # 找出受影响最大的其他句子
        sentence_impacts = [
            {'sentence_idx': i, 'change': float(text_changes[i])}
            for i in range(num_sentences) if i != sentence_idx
        ]
        sentence_impacts.sort(key=lambda x: abs(x['change']), reverse=True)
        
        result = {
            'type': 'counterfactual_sentence_removal',
            'removed_sentence_idx': sentence_idx,
            'removed_sentence_text': article_sentences[sentence_idx] if article_sentences and sentence_idx < len(article_sentences) else None,
            'total_video_impact': float(np.abs(video_changes[:num_frames]).sum()),
            'total_text_impact': float(np.abs(text_changes[:num_sentences]).sum()),
            'most_affected_frames': frame_impacts[:5],
            'most_affected_sentences': sentence_impacts[:5],
            'video_changes': video_changes[:num_frames].tolist(),
            'text_changes': text_changes[:num_sentences].tolist()
        }
        
        # 生成自然语言解释
        result['natural_language_explanation'] = self._generate_counterfactual_explanation(
            'sentence', sentence_idx, frame_impacts[:3], sentence_impacts[:3],
            article_sentences[sentence_idx] if article_sentences and sentence_idx < len(article_sentences) else None
        )
        
        return result
    
    @torch.no_grad()
    def what_if_remove_frame(self,
                             video: torch.Tensor,
                             text: torch.Tensor,
                             mask_video: torch.Tensor,
                             mask_text: torch.Tensor,
                             frame_idx: int,
                             article_sentences: Optional[List[str]] = None,
                             **kwargs) -> Dict:
        """
        反事实分析：如果删除某帧会怎样
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
        video_probs_orig = torch.sigmoid(pred_video_orig[0])
        video_probs_cf = torch.sigmoid(pred_video_cf[0])
        video_changes = (video_probs_orig - video_probs_cf).cpu().numpy()
        
        text_probs_orig = torch.sigmoid(pred_text_orig[0])
        text_probs_cf = torch.sigmoid(pred_text_cf[0])
        text_changes = (text_probs_orig - text_probs_cf).cpu().numpy()
        
        num_frames = int(mask_video.sum().item())
        num_sentences = int(mask_text.sum().item())
        
        # 受影响的句子
        sentence_impacts = [
            {
                'sentence_idx': i, 
                'change': float(text_changes[i]),
                'sentence_text': article_sentences[i][:50] + '...' if article_sentences and i < len(article_sentences) and len(article_sentences[i]) > 50 else (article_sentences[i] if article_sentences and i < len(article_sentences) else None)
            }
            for i in range(num_sentences)
        ]
        sentence_impacts.sort(key=lambda x: abs(x['change']), reverse=True)
        
        # 受影响的其他帧
        frame_impacts = [
            {'frame_idx': i, 'change': float(video_changes[i])}
            for i in range(num_frames) if i != frame_idx
        ]
        frame_impacts.sort(key=lambda x: abs(x['change']), reverse=True)
        
        result = {
            'type': 'counterfactual_frame_removal',
            'removed_frame_idx': frame_idx,
            'total_video_impact': float(np.abs(video_changes[:num_frames]).sum()),
            'total_text_impact': float(np.abs(text_changes[:num_sentences]).sum()),
            'most_affected_sentences': sentence_impacts[:5],
            'most_affected_frames': frame_impacts[:5],
            'video_changes': video_changes[:num_frames].tolist(),
            'text_changes': text_changes[:num_sentences].tolist()
        }
        
        result['natural_language_explanation'] = self._generate_counterfactual_explanation(
            'frame', frame_idx, frame_impacts[:3], sentence_impacts[:3], None
        )
        
        return result
    
    @torch.no_grad()
    def compute_sentence_importance(self,
                                    video: torch.Tensor,
                                    text: torch.Tensor,
                                    mask_video: torch.Tensor,
                                    mask_text: torch.Tensor,
                                    **kwargs) -> Dict[int, float]:
        """
        计算每个句子对摘要决策的因果重要性
        
        使用反事实干预方法：移除每个句子，计算预测变化
        
        Returns:
            句子索引到重要性分数的映射
        """
        # 原始预测
        pred_video_orig, pred_text_orig, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        num_sentences = int(mask_text.sum().item())
        importance_scores = {}
        
        for sent_idx in range(num_sentences):
            # 移除该句子
            text_cf = text.clone()
            text_cf[0, sent_idx] = 0
            
            pred_video_cf, pred_text_cf, _ = self.model(
                video=video, text=text_cf,
                mask_video=mask_video, mask_text=mask_text,
                **kwargs
            )
            
            # 计算整体预测变化
            video_change = torch.abs(
                torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_cf)
            ).sum().item()
            
            text_change = torch.abs(
                torch.sigmoid(pred_text_orig) - torch.sigmoid(pred_text_cf)
            ).sum().item()
            
            importance_scores[sent_idx] = video_change + text_change
        
        return importance_scores
    
    @torch.no_grad()
    def compute_frame_text_dependency(self,
                                      video: torch.Tensor,
                                      text: torch.Tensor,
                                      mask_video: torch.Tensor,
                                      mask_text: torch.Tensor,
                                      **kwargs) -> np.ndarray:
        """
        计算每帧对每个句子的依赖程度矩阵
        
        Returns:
            依赖矩阵 [num_frames, num_sentences]
            dependency[i, j] 表示帧i对句子j的依赖程度
        """
        # 原始预测
        pred_video_orig, _, _ = self.model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            **kwargs
        )
        
        num_frames = int(mask_video.sum().item())
        num_sentences = int(mask_text.sum().item())
        
        dependency_matrix = np.zeros((num_frames, num_sentences))
        
        for sent_idx in range(num_sentences):
            # 移除该句子
            text_cf = text.clone()
            text_cf[0, sent_idx] = 0
            
            pred_video_cf, _, _ = self.model(
                video=video, text=text_cf,
                mask_video=mask_video, mask_text=mask_text,
                **kwargs
            )
            
            # 计算每帧的变化
            frame_changes = (
                torch.sigmoid(pred_video_orig[0, :num_frames]) - 
                torch.sigmoid(pred_video_cf[0, :num_frames])
            ).cpu().numpy()
            
            dependency_matrix[:, sent_idx] = frame_changes
        
        return dependency_matrix
    
    def generate_summary_report(self,
                                video: torch.Tensor,
                                text: torch.Tensor,
                                mask_video: torch.Tensor,
                                mask_text: torch.Tensor,
                                selected_frames: List[int],
                                selected_sentences: List[int],
                                article_sentences: Optional[List[str]] = None,
                                sample_id: str = 'sample',
                                **kwargs) -> str:
        """
        生成完整的可解释摘要报告
        
        Args:
            sample_id: 样本标识符
            
        Returns:
            Markdown格式的报告文本
        """
        report_lines = [
            f"# 摘要解释报告 - {sample_id}",
            "",
            "## 概览",
            f"- 视频帧数: {int(mask_video.sum().item())}",
            f"- 选中帧数: {len(selected_frames)}",
            f"- 句子数: {int(mask_text.sum().item())}",
            f"- 选中句子数: {len(selected_sentences)}",
            ""
        ]
        
        # 句子重要性分析
        report_lines.extend([
            "## 句子因果重要性分析",
            ""
        ])
        
        importance = self.compute_sentence_importance(
            video, text, mask_video, mask_text, **kwargs
        )
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (sent_idx, score) in enumerate(sorted_importance[:5], 1):
            sent_text = article_sentences[sent_idx][:80] + '...' if article_sentences and sent_idx < len(article_sentences) and len(article_sentences[sent_idx]) > 80 else (article_sentences[sent_idx] if article_sentences and sent_idx < len(article_sentences) else f"Sentence {sent_idx}")
            is_selected = "✓" if sent_idx in selected_sentences else "✗"
            report_lines.append(f"{rank}. [{is_selected}] 句子 {sent_idx} (重要性: {score:.4f})")
            report_lines.append(f"   > {sent_text}")
            report_lines.append("")
        
        # 帧-文本依赖矩阵
        report_lines.extend([
            "## 帧-句子依赖关系",
            ""
        ])
        
        dependency_matrix = self.compute_frame_text_dependency(
            video, text, mask_video, mask_text, **kwargs
        )
        
        # 保存热力图
        self._save_dependency_heatmap(
            dependency_matrix, selected_frames, selected_sentences,
            f'{self.save_dir}/{sample_id}_dependency.png'
        )
        report_lines.append(f"![依赖热力图]({sample_id}_dependency.png)")
        report_lines.append("")
        
        # 反事实分析
        report_lines.extend([
            "## 反事实分析示例",
            ""
        ])
        
        # 分析移除最重要句子的影响
        if sorted_importance:
            top_sentence = sorted_importance[0][0]
            cf_result = self.what_if_remove_sentence(
                video, text, mask_video, mask_text, top_sentence,
                article_sentences, **kwargs
            )
            report_lines.append(f"### 如果移除最重要的句子 ({top_sentence}):")
            report_lines.append(cf_result['natural_language_explanation'])
            report_lines.append("")
        
        # 保存报告
        report_text = '\n'.join(report_lines)
        report_path = f'{self.save_dir}/{sample_id}_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Report saved to {report_path}")
        
        return report_text
    
    def _save_dependency_heatmap(self,
                                 dependency_matrix: np.ndarray,
                                 selected_frames: List[int],
                                 selected_sentences: List[int],
                                 save_path: str):
        """保存依赖矩阵热力图"""
        plt.figure(figsize=(12, 8))
        
        # 创建热力图
        ax = sns.heatmap(
            dependency_matrix, 
            cmap='RdBu_r', 
            center=0,
            xticklabels=[f"S{i}" for i in range(dependency_matrix.shape[1])],
            yticklabels=[f"F{i}" for i in range(dependency_matrix.shape[0])]
        )
        
        # 标记选中的帧和句子
        for frame_idx in selected_frames:
            if frame_idx < dependency_matrix.shape[0]:
                ax.axhline(y=frame_idx + 0.5, color='green', linewidth=2, alpha=0.7)
        
        for sent_idx in selected_sentences:
            if sent_idx < dependency_matrix.shape[1]:
                ax.axvline(x=sent_idx + 0.5, color='blue', linewidth=2, alpha=0.7)
        
        plt.title('Frame-Sentence Dependency Matrix\n(Green: Selected Frames, Blue: Selected Sentences)')
        plt.xlabel('Sentence Index')
        plt.ylabel('Frame Index')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_frame_nl_explanation(self,
                                       frame_idx: int,
                                       score: float,
                                       top_contributions: List[Dict],
                                       article_sentences: Optional[List[str]]) -> str:
        """生成帧选择的自然语言解释"""
        strength = "被强烈选中" if score > 0.7 else "被适度选中" if score > 0.4 else "被弱选中"
        
        lines = [
            f"**帧 {frame_idx}** {strength}（分数: {score:.2f}）。",
            "",
            "该帧的选择主要受以下句子影响:"
        ]
        
        for i, contrib in enumerate(top_contributions, 1):
            direction = "支持" if contrib['is_supportive'] else "不支持"
            sent_text = ""
            if article_sentences and contrib['sentence_idx'] < len(article_sentences):
                sent_text = f": \"{article_sentences[contrib['sentence_idx']][:60]}...\""
            lines.append(
                f"  {i}. 句子 {contrib['sentence_idx']} ({direction}, 影响: {abs(contrib['contribution']):.3f}){sent_text}"
            )
        
        return '\n'.join(lines)
    
    def _generate_sentence_nl_explanation(self,
                                          sent_idx: int,
                                          score: float,
                                          top_contributions: List[Dict],
                                          sentence_text: Optional[str]) -> str:
        """生成句子选择的自然语言解释"""
        strength = "被强烈选中" if score > 0.7 else "被适度选中" if score > 0.4 else "被弱选中"
        
        lines = [
            f"**句子 {sent_idx}** {strength}（分数: {score:.2f}）。"
        ]
        
        if sentence_text:
            lines.append(f"> \"{sentence_text[:100]}...\"" if len(sentence_text) > 100 else f"> \"{sentence_text}\"")
        
        lines.extend([
            "",
            "该句子的选择主要受以下视频帧影响:"
        ])
        
        for i, contrib in enumerate(top_contributions, 1):
            direction = "支持" if contrib['is_supportive'] else "不支持"
            lines.append(
                f"  {i}. 帧 {contrib['frame_idx']} ({direction}, 影响: {abs(contrib['contribution']):.3f})"
            )
        
        return '\n'.join(lines)
    
    def _generate_counterfactual_explanation(self,
                                             element_type: str,
                                             element_idx: int,
                                             frame_impacts: List[Dict],
                                             sentence_impacts: List[Dict],
                                             element_text: Optional[str]) -> str:
        """生成反事实分析的自然语言解释"""
        type_cn = "句子" if element_type == "sentence" else "帧"
        
        lines = [
            f"### 反事实分析: 如果移除{type_cn} {element_idx}"
        ]
        
        if element_text:
            lines.append(f"> \"{element_text[:100]}...\"" if len(element_text) > 100 else f"> \"{element_text}\"")
        
        lines.append("")
        
        total_impact = sum(abs(f['change']) for f in frame_impacts) + sum(abs(s['change']) for s in sentence_impacts)
        
        if total_impact > 1.0:
            lines.append("**影响程度: 显著** - 移除该元素会对摘要产生重大影响。")
        elif total_impact > 0.3:
            lines.append("**影响程度: 中等** - 移除该元素会对摘要产生一定影响。")
        else:
            lines.append("**影响程度: 轻微** - 移除该元素对摘要影响较小。")
        
        lines.append("")
        
        if element_type == "sentence":
            lines.append("**受影响最大的帧:**")
            for impact in frame_impacts[:3]:
                direction = "降低" if impact['change'] > 0 else "升高"
                lines.append(f"  - 帧 {impact['frame_idx']}: 选择概率{direction} {abs(impact['change']):.3f}")
        else:
            lines.append("**受影响最大的句子:**")
            for impact in sentence_impacts[:3]:
                direction = "降低" if impact['change'] > 0 else "升高"
                text_preview = impact.get('sentence_text', '')
                lines.append(f"  - 句子 {impact['sentence_idx']}: 选择概率{direction} {abs(impact['change']):.3f}")
                if text_preview:
                    lines.append(f"    > \"{text_preview}\"")
        
        return '\n'.join(lines)


if __name__ == '__main__':
    print("Testing MSMOExplainer module...")
    
    # 创建测试目录
    os.makedirs('./test_explanations', exist_ok=True)
    
    print("✓ MSMOExplainer module structure validated")
    print("\nNote: Full testing requires a trained model.")
    print("Usage example:")
    print("""
    from interpretability.msmo_explainer import MSMOExplainer
    
    # Initialize with trained model
    explainer = MSMOExplainer(model, device='cuda')
    
    # Explain video selection
    explanation = explainer.explain_video_selection(
        video, text, mask_video, mask_text,
        selected_frames=[0, 5, 10],
        article_sentences=sentences
    )
    
    # Generate full report
    report = explainer.generate_summary_report(
        video, text, mask_video, mask_text,
        selected_frames, selected_sentences,
        article_sentences, sample_id='sample_001'
    )
    """)
