"""
Causal Explainer for Interpretability Analysis
Provides tools for computing causal effects and generating explanations
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class CausalEffectAnalyzer:
    """
    Analyze causal effects using counterfactual interventions
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained video summarization model
        """
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def compute_ATE(self,
                    video: torch.Tensor,
                    text_orig: torch.Tensor,
                    text_cf: torch.Tensor,
                    mask_video: torch.Tensor,
                    mask_text: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        Compute Average Treatment Effect (ATE)
        
        ATE = E[Y | do(X=x')] - E[Y | do(X=x)]
        
        Args:
            video: Video features [B, T_v, D_v]
            text_orig: Original text [B, T_t, D_t]
            text_cf: Counterfactual text [B, T_t, D_t]
            mask_video: Video mask [B, T_v]
            mask_text: Text mask [B, T_t]
            
        Returns:
            Dictionary with ATE metrics
        """
        # Predict with original text
        outputs_orig = self.model(
            video=video,
            text=text_orig,
            mask_video=mask_video,
            mask_text=mask_text,
            **kwargs
        )
        # Handle both tuple return (Model_MSMO/BLiSS) and dict return formats
        if isinstance(outputs_orig, tuple):
            pred_orig = outputs_orig[0]  # pred_video is first element
        else:
            pred_orig = outputs_orig['video_pred_cls']
        
        # Predict with counterfactual text
        outputs_cf = self.model(
            video=video,
            text=text_cf,
            mask_video=mask_video,
            mask_text=mask_text,
            **kwargs
        )
        if isinstance(outputs_cf, tuple):
            pred_cf = outputs_cf[0]  # pred_video is first element
        else:
            pred_cf = outputs_cf['video_pred_cls']
        
        # Compute ATE
        prob_orig = torch.sigmoid(pred_orig)
        prob_cf = torch.sigmoid(pred_cf)
        
        # Average over valid positions
        valid_mask = mask_video > 0
        ate = (prob_orig[valid_mask] - prob_cf[valid_mask]).mean().item()
        
        # Additional metrics
        ate_std = (prob_orig[valid_mask] - prob_cf[valid_mask]).std().item()
        ate_abs = torch.abs(prob_orig[valid_mask] - prob_cf[valid_mask]).mean().item()
        
        return {
            'ATE': ate,
            'ATE_std': ate_std,
            'ATE_abs': ate_abs,
            'effect_size': ate_abs / (prob_orig[valid_mask].std().item() + 1e-8)
        }
    
    @torch.no_grad()
    def compute_CATE(self,
                     video: torch.Tensor,
                     text_orig: torch.Tensor,
                     text_cf: torch.Tensor,
                     mask_video: torch.Tensor,
                     mask_text: torch.Tensor,
                     condition_indices: torch.Tensor,
                     **kwargs) -> Dict[str, float]:
        """
        Compute Conditional Average Treatment Effect (CATE)
        
        CATE = E[Y | do(X=x'), C=c] - E[Y | do(X=x), C=c]
        
        Args:
            condition_indices: Indices of conditional variables [B]
            
        Returns:
            Dictionary with CATE metrics
        """
        # Get predictions
        outputs_orig = self.model(video=video, text=text_orig, 
                                 mask_video=mask_video, mask_text=mask_text, **kwargs)
        outputs_cf = self.model(video=video, text=text_cf,
                               mask_video=mask_video, mask_text=mask_text, **kwargs)
        
        # Handle both tuple return (Model_MSMO/BLiSS) and dict return formats
        if isinstance(outputs_orig, tuple):
            pred_orig = outputs_orig[0]
        else:
            pred_orig = outputs_orig['video_pred_cls']
        
        if isinstance(outputs_cf, tuple):
            pred_cf = outputs_cf[0]
        else:
            pred_cf = outputs_cf['video_pred_cls']
        
        prob_orig = torch.sigmoid(pred_orig)
        prob_cf = torch.sigmoid(pred_cf)
        
        # Compute CATE for each condition
        cate_dict = {}
        for cond_val in torch.unique(condition_indices):
            cond_mask = (condition_indices == cond_val).unsqueeze(1) & (mask_video > 0)
            if cond_mask.sum() > 0:
                cate = (prob_orig[cond_mask] - prob_cf[cond_mask]).mean().item()
                cate_dict[f'CATE_cond_{int(cond_val)}'] = cate
        
        return cate_dict
    
    def identify_key_interventions(self,
                                   video: torch.Tensor,
                                   text: torch.Tensor,
                                   mask_video: torch.Tensor,
                                   mask_text: torch.Tensor,
                                   num_top: int = 5,
                                   **kwargs) -> List[Tuple[int, float]]:
        """
        Identify which text positions have the largest causal effect
        
        Args:
            num_top: Number of top interventions to return
            
        Returns:
            List of (position, effect_magnitude) tuples
        """
        B, T_text, D = text.shape
        effects = []
        
        with torch.no_grad():
            # Base prediction
            outputs_base = self.model(video=video, text=text,
                                     mask_video=mask_video, mask_text=mask_text, **kwargs)
            # Handle both tuple return (Model_MSMO/BLiSS) and dict return formats
            if isinstance(outputs_base, tuple):
                pred_base = outputs_base[0]
            else:
                pred_base = outputs_base['video_pred_cls']
            prob_base = torch.sigmoid(pred_base)
            
            # Test each text position
            for t in range(T_text):
                if mask_text[0, t] == 0:  # Skip padding
                    continue
                
                # Create counterfactual by zeroing out position t
                text_cf = text.clone()
                text_cf[:, t, :] = 0
                
                outputs_cf = self.model(video=video, text=text_cf,
                                       mask_video=mask_video, mask_text=mask_text, **kwargs)
                if isinstance(outputs_cf, tuple):
                    pred_cf = outputs_cf[0]
                else:
                    pred_cf = outputs_cf['video_pred_cls']
                prob_cf = torch.sigmoid(pred_cf)
                
                # Compute effect magnitude
                effect = torch.abs(prob_base - prob_cf).mean().item()
                effects.append((t, effect))
        
        # Sort by effect magnitude
        effects.sort(key=lambda x: x[1], reverse=True)
        return effects[:num_top]
    
    @torch.no_grad()
    def compute_sentence_importance(self,
                                    video: torch.Tensor,
                                    text: torch.Tensor,
                                    mask_video: torch.Tensor,
                                    mask_text: torch.Tensor,
                                    **kwargs) -> Dict[int, float]:
        """
        计算每个句子对摘要决策的因果重要性 (MSMO专用)
        
        使用反事实干预方法：移除每个句子，计算预测变化
        
        Args:
            video: Video features [B, T_v, D_v]
            text: Text features [B, T_t, D_t]
            mask_video: Video mask
            mask_text: Text mask
            
        Returns:
            Dictionary mapping sentence index to importance score
        """
        B, T_text, D = text.shape
        
        # Base prediction
        outputs_base = self.model(video=video, text=text,
                                 mask_video=mask_video, mask_text=mask_text, **kwargs)
        pred_video_base = outputs_base[0] if isinstance(outputs_base, tuple) else outputs_base.get('video_pred_cls', outputs_base)
        pred_text_base = outputs_base[1] if isinstance(outputs_base, tuple) else outputs_base.get('text_pred_cls', outputs_base)
        
        prob_video_base = torch.sigmoid(pred_video_base)
        prob_text_base = torch.sigmoid(pred_text_base)
        
        importance_scores = {}
        num_sentences = int(mask_text.sum().item()) if mask_text.dim() == 1 else int(mask_text[0].sum().item())
        
        for sent_idx in range(num_sentences):
            # Create counterfactual by zeroing out sentence
            text_cf = text.clone()
            text_cf[:, sent_idx, :] = 0
            
            outputs_cf = self.model(video=video, text=text_cf,
                                   mask_video=mask_video, mask_text=mask_text, **kwargs)
            pred_video_cf = outputs_cf[0] if isinstance(outputs_cf, tuple) else outputs_cf.get('video_pred_cls', outputs_cf)
            pred_text_cf = outputs_cf[1] if isinstance(outputs_cf, tuple) else outputs_cf.get('text_pred_cls', outputs_cf)
            
            prob_video_cf = torch.sigmoid(pred_video_cf)
            prob_text_cf = torch.sigmoid(pred_text_cf)
            
            # Importance = total prediction change
            video_change = torch.abs(prob_video_base - prob_video_cf).sum().item()
            text_change = torch.abs(prob_text_base - prob_text_cf).sum().item()
            
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
        计算每帧对每个句子的依赖程度矩阵 (MSMO专用)
        
        Returns:
            Dependency matrix [num_frames, num_sentences]
            dependency[i, j] = how much frame i depends on sentence j
        """
        import numpy as np
        
        # Base prediction
        outputs_base = self.model(video=video, text=text,
                                 mask_video=mask_video, mask_text=mask_text, **kwargs)
        pred_video_base = outputs_base[0] if isinstance(outputs_base, tuple) else outputs_base.get('video_pred_cls', outputs_base)
        
        num_frames = int(mask_video.sum().item()) if mask_video.dim() == 1 else int(mask_video[0].sum().item())
        num_sentences = int(mask_text.sum().item()) if mask_text.dim() == 1 else int(mask_text[0].sum().item())
        
        dependency_matrix = np.zeros((num_frames, num_sentences))
        
        for sent_idx in range(num_sentences):
            # Remove sentence
            text_cf = text.clone()
            text_cf[:, sent_idx, :] = 0
            
            outputs_cf = self.model(video=video, text=text_cf,
                                   mask_video=mask_video, mask_text=mask_text, **kwargs)
            pred_video_cf = outputs_cf[0] if isinstance(outputs_cf, tuple) else outputs_cf.get('video_pred_cls', outputs_cf)
            
            # Change in each frame's prediction
            frame_changes = (
                torch.sigmoid(pred_video_base[0, :num_frames]) - 
                torch.sigmoid(pred_video_cf[0, :num_frames])
            ).cpu().numpy()
            
            dependency_matrix[:, sent_idx] = frame_changes
        
        return dependency_matrix
    
    def generate_explanation_report(self,
                                    video: torch.Tensor,
                                    text: torch.Tensor,
                                    mask_video: torch.Tensor,
                                    mask_text: torch.Tensor,
                                    selected_frames: List[int],
                                    selected_sentences: List[int],
                                    article_sentences: Optional[List[str]] = None,
                                    **kwargs) -> str:
        """
        生成人类可读的摘要解释报告 (MSMO专用)
        
        Args:
            selected_frames: List of selected frame indices
            selected_sentences: List of selected sentence indices
            article_sentences: Optional list of original sentence texts
            
        Returns:
            Markdown formatted explanation report
        """
        report_lines = [
            "# Summarization Explanation Report",
            "",
            "## Summary Statistics",
            f"- Total frames: {int(mask_video.sum().item())}",
            f"- Selected frames: {len(selected_frames)}",
            f"- Total sentences: {int(mask_text.sum().item())}",
            f"- Selected sentences: {len(selected_sentences)}",
            ""
        ]
        
        # Sentence importance
        importance = self.compute_sentence_importance(
            video, text, mask_video, mask_text, **kwargs
        )
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        report_lines.extend([
            "## Sentence Importance Ranking",
            ""
        ])
        
        for rank, (sent_idx, score) in enumerate(sorted_importance[:5], 1):
            is_selected = "✓" if sent_idx in selected_sentences else "✗"
            sent_preview = ""
            if article_sentences and sent_idx < len(article_sentences):
                sent_preview = f': "{article_sentences[sent_idx][:60]}..."'
            report_lines.append(f"{rank}. [{is_selected}] Sentence {sent_idx} (importance: {score:.4f}){sent_preview}")
        
        report_lines.append("")
        
        # Key interventions
        interventions = self.identify_key_interventions(
            video, text, mask_video, mask_text, num_top=5, **kwargs
        )
        
        report_lines.extend([
            "## Key Text Positions (by causal effect)",
            ""
        ])
        
        for rank, (pos, effect) in enumerate(interventions, 1):
            report_lines.append(f"{rank}. Position {pos}: effect magnitude = {effect:.4f}")
        
        return '\n'.join(report_lines)


class CausalVisualizer:
    """
    Visualize causal effects and explanations
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_causal_heatmap(self,
                           attention_orig: np.ndarray,
                           attention_cf: np.ndarray,
                           save_name: str = 'causal_attention.png'):
        """
        Plot comparison of attention patterns
        
        Args:
            attention_orig: Original attention [T_v, T_t]
            attention_cf: Counterfactual attention [T_v, T_t]
            save_name: Output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original attention
        sns.heatmap(attention_orig, ax=axes[0], cmap='viridis', cbar=True)
        axes[0].set_title('Original Attention')
        axes[0].set_xlabel('Text Position')
        axes[0].set_ylabel('Video Frame')
        
        # Counterfactual attention
        sns.heatmap(attention_cf, ax=axes[1], cmap='viridis', cbar=True)
        axes[1].set_title('Counterfactual Attention')
        axes[1].set_xlabel('Text Position')
        axes[1].set_ylabel('Video Frame')
        
        # Difference
        diff = attention_orig - attention_cf
        sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', center=0, cbar=True)
        axes[2].set_title('Causal Effect (Difference)')
        axes[2].set_xlabel('Text Position')
        axes[2].set_ylabel('Video Frame')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap to {self.save_dir}/{save_name}")
    
    def plot_feature_space(self,
                          feat_orig: np.ndarray,
                          feat_cf_min: np.ndarray,
                          feat_cf_sev: np.ndarray,
                          save_name: str = 'feature_space.png'):
        """
        Plot t-SNE visualization of feature space
        
        Args:
            feat_orig: Original features [N, D]
            feat_cf_min: Minimal CF features [N, D]
            feat_cf_sev: Severe CF features [N, D]
        """
        from sklearn.manifold import TSNE
        
        # Combine features
        all_features = np.vstack([feat_orig, feat_cf_min, feat_cf_sev])
        labels = np.array(['Original'] * len(feat_orig) + 
                         ['Minimal CF'] * len(feat_cf_min) +
                         ['Severe CF'] * len(feat_cf_sev))
        
        # Check if we have enough samples for t-SNE
        n_samples = len(all_features)
        if n_samples < 10:
            print(f"⚠️ Not enough samples for t-SNE ({n_samples} < 10). Skipping visualization.")
            return
        
        # Adjust perplexity based on sample size (must be less than n_samples)
        perplexity = min(30, n_samples // 3)
        perplexity = max(5, perplexity)  # Minimum perplexity of 5
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(all_features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        for label in ['Original', 'Minimal CF', 'Severe CF']:
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       label=label, alpha=0.6, s=50)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Feature Space Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{self.save_dir}/{save_name}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved t-SNE plot to {self.save_dir}/{save_name}")
    
    def generate_explanation_text(self,
                                 text_orig: List[str],
                                 text_cf: List[str],
                                 pred_orig: float,
                                 pred_cf: float,
                                 intervention_type: str) -> str:
        """
        Generate natural language explanation
        
        Args:
            text_orig: Original text sentences
            text_cf: Counterfactual text sentences
            pred_orig: Original prediction score
            pred_cf: Counterfactual prediction score
            intervention_type: Type of intervention applied
            
        Returns:
            Natural language explanation
        """
        effect = pred_orig - pred_cf
        direction = "increased" if effect > 0 else "decreased"
        
        explanation = f"""
Causal Explanation:
-------------------
Intervention Type: {intervention_type}

Original Prediction: {pred_orig:.4f}
Counterfactual Prediction: {pred_cf:.4f}
Causal Effect: {abs(effect):.4f} ({direction})

Analysis:
When the text was modified from:
  "{' '.join(text_orig[:3])}..."
To:
  "{' '.join(text_cf[:3])}..."

The model's confidence in selecting video frames {direction} by {abs(effect):.2%}.
This indicates that the model is {'strongly' if abs(effect) > 0.3 else 'moderately' if abs(effect) > 0.1 else 'weakly'} 
influenced by this textual information.
        """
        
        return explanation.strip()


if __name__ == '__main__':
    print("Testing CausalExplainer...")
    
    # Test with dummy data
    visualizer = CausalVisualizer(save_dir='./test_viz')
    
    # Create dummy attention patterns
    attention_orig = np.random.rand(20, 15)
    attention_cf = attention_orig + np.random.randn(20, 15) * 0.2
    
    visualizer.plot_causal_heatmap(attention_orig, attention_cf, 'test_heatmap.png')
    
    # Test feature space visualization
    feat_orig = np.random.randn(50, 128)
    feat_cf_min = feat_orig + np.random.randn(50, 128) * 0.1
    feat_cf_sev = np.random.randn(50, 128)
    
    visualizer.plot_feature_space(feat_orig, feat_cf_min, feat_cf_sev, 'test_tsne.png')
    
    print("✓ Visualization tests passed!")
