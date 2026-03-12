"""
Causal Evaluation Metrics
Metrics for evaluating causal models and counterfactual quality
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import spearmanr, kendalltau


def counterfactual_validity(text_orig: torch.Tensor,
                            text_cf: torch.Tensor,
                            pred_orig: torch.Tensor,
                            pred_cf: torch.Tensor,
                            semantic_sim_threshold: float = 0.7) -> Dict[str, float]:
    """
    Measure counterfactual validity:
    Good CF should be semantically similar but produce different predictions
    
    CFV = semantic_similarity(x, x_cf) * (1 - task_similarity(y, y_cf))
    
    Args:
        text_orig: Original text features [B, T, D]
        text_cf: Counterfactual text features [B, T, D]
        pred_orig: Original predictions [B, T]
        pred_cf: Counterfactual predictions [B, T]
        semantic_sim_threshold: Threshold for semantic similarity
        
    Returns:
        Dictionary with validity metrics
    """
    # Compute semantic similarity (cosine)
    text_orig_flat = text_orig.reshape(-1, text_orig.shape[-1])
    text_cf_flat = text_cf.reshape(-1, text_cf.shape[-1])
    
    semantic_sim = torch.cosine_similarity(text_orig_flat, text_cf_flat).mean().item()
    
    # Compute task similarity (prediction difference)
    prob_orig = torch.sigmoid(pred_orig)
    prob_cf = torch.sigmoid(pred_cf)
    task_diff = torch.abs(prob_orig - prob_cf).mean().item()
    
    # CFV score
    cfv_score = semantic_sim * task_diff
    
    # Quality check
    is_valid = (semantic_sim >= semantic_sim_threshold) and (task_diff >= 0.1)
    
    return {
        'CFV_score': cfv_score,
        'semantic_similarity': semantic_sim,
        'task_difference': task_diff,
        'is_valid': float(is_valid)
    }


def causal_sensitivity(intervention_magnitude: torch.Tensor,
                      prediction_change: torch.Tensor) -> Dict[str, float]:
    """
    Measure model's sensitivity to causal interventions
    
    CS = ||Δy|| / ||Δx||
    
    Args:
        intervention_magnitude: Magnitude of intervention [B]
        prediction_change: Change in predictions [B]
        
    Returns:
        Causal sensitivity metrics
    """
    # Avoid division by zero
    epsilon = 1e-8
    sensitivity = (prediction_change / (intervention_magnitude + epsilon)).mean().item()
    
    # Correlation between intervention and effect
    if len(intervention_magnitude) > 1:
        corr, _ = spearmanr(intervention_magnitude.cpu().numpy(), 
                           prediction_change.cpu().numpy())
    else:
        corr = 0.0
    
    return {
        'sensitivity': sensitivity,
        'correlation': corr,
        'avg_intervention': intervention_magnitude.mean().item(),
        'avg_effect': prediction_change.mean().item()
    }


def feature_disentanglement_score(causal_feat: torch.Tensor,
                                  spurious_feat: torch.Tensor) -> Dict[str, float]:
    """
    Measure separation between causal and spurious features
    
    FDS = 1 - MI(z_causal, z_spurious) / H(z)
    Approximated using correlation
    
    Args:
        causal_feat: Causal features [B, T, D//2]
        spurious_feat: Spurious features [B, T, D//2]
        
    Returns:
        Disentanglement metrics
    """
    # Flatten
    B, T, D = causal_feat.shape
    causal_flat = causal_feat.reshape(-1, D)
    spurious_flat = spurious_feat.reshape(-1, D)
    
    # Normalize
    causal_norm = torch.nn.functional.normalize(causal_flat, p=2, dim=1)
    spurious_norm = torch.nn.functional.normalize(spurious_flat, p=2, dim=1)
    
    # Compute correlation matrix
    correlation = torch.mm(causal_norm.t(), spurious_norm)  # [D, D]
    
    # Metrics
    avg_abs_corr = torch.abs(correlation).mean().item()
    max_corr = torch.abs(correlation).max().item()
    
    # FDS: lower correlation = better disentanglement
    fds = 1.0 - avg_abs_corr
    
    return {
        'FDS': fds,
        'avg_correlation': avg_abs_corr,
        'max_correlation': max_corr,
        'is_disentangled': float(avg_abs_corr < 0.3)
    }


def explanation_fidelity(model_importance: np.ndarray,
                        human_importance: np.ndarray) -> Dict[str, float]:
    """
    Measure agreement between model explanations and human judgments
    
    Args:
        model_importance: Model-assigned importance scores [N]
        human_importance: Human-annotated importance scores [N]
        
    Returns:
        Fidelity metrics
    """
    # Rank correlation
    spearman_corr, spearman_p = spearmanr(model_importance, human_importance)
    kendall_corr, kendall_p = kendalltau(model_importance, human_importance)
    
    # Top-k overlap
    k = max(3, len(model_importance) // 5)
    model_topk = set(np.argsort(model_importance)[-k:])
    human_topk = set(np.argsort(human_importance)[-k:])
    topk_overlap = len(model_topk & human_topk) / k
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'kendall_correlation': kendall_corr,
        'topk_overlap': topk_overlap,
        'fidelity_score': (spearman_corr + topk_overlap) / 2
    }


def compute_ate_metrics(pred_orig: torch.Tensor,
                       pred_cf: torch.Tensor,
                       mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute Average Treatment Effect (ATE) metrics
    
    Args:
        pred_orig: Original predictions [B, T]
        pred_cf: Counterfactual predictions [B, T]
        mask: Valid positions [B, T]
        
    Returns:
        ATE metrics
    """
    prob_orig = torch.sigmoid(pred_orig)
    prob_cf = torch.sigmoid(pred_cf)
    
    valid_mask = mask > 0
    
    if valid_mask.sum() == 0:
        return {'ATE': 0.0, 'ATE_abs': 0.0, 'ATE_std': 0.0}
    
    # Effect per sample
    effects = prob_orig[valid_mask] - prob_cf[valid_mask]
    
    ate = effects.mean().item()
    ate_abs = torch.abs(effects).mean().item()
    ate_std = effects.std().item()
    
    # Effect size (standardized)
    pooled_std = torch.cat([prob_orig[valid_mask], prob_cf[valid_mask]]).std().item()
    effect_size = ate / (pooled_std + 1e-8)
    
    return {
        'ATE': ate,
        'ATE_abs': ate_abs,
        'ATE_std': ate_std,
        'effect_size': effect_size
    }


class CausalMetricsTracker:
    """
    Track causal metrics during training/evaluation
    """
    
    def __init__(self):
        self.metrics_history = {
            'CFV': [],
            'sensitivity': [],
            'FDS': [],
            'ATE': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """Add metrics for current batch"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        """Get average metrics across all batches"""
        return {key: np.mean(values) if values else 0.0 
                for key, values in self.metrics_history.items()}
    
    def reset(self):
        """Reset all metrics"""
        for key in self.metrics_history:
            self.metrics_history[key] = []
    
    def summary(self) -> str:
        """Generate summary string"""
        avg_metrics = self.get_average()
        summary = "Causal Metrics Summary:\n"
        summary += "-" * 40 + "\n"
        for key, value in avg_metrics.items():
            summary += f"{key:20s}: {value:.4f}\n"
        return summary


if __name__ == '__main__':
    print("Testing causal metrics...")
    
    # Test CFV
    text_orig = torch.randn(4, 10, 512)
    text_cf = text_orig + torch.randn(4, 10, 512) * 0.2
    pred_orig = torch.randn(4, 10)
    pred_cf = pred_orig + torch.randn(4, 10) * 0.5
    
    cfv = counterfactual_validity(text_orig, text_cf, pred_orig, pred_cf)
    print("\nCounterfactual Validity:")
    for k, v in cfv.items():
        print(f"  {k}: {v:.4f}")
    
    # Test sensitivity
    intervention_mag = torch.rand(10)
    pred_change = torch.rand(10)
    cs = causal_sensitivity(intervention_mag, pred_change)
    print("\nCausal Sensitivity:")
    for k, v in cs.items():
        print(f"  {k}: {v:.4f}")
    
    # Test FDS
    causal_feat = torch.randn(4, 10, 256)
    spurious_feat = torch.randn(4, 10, 256)
    fds = feature_disentanglement_score(causal_feat, spurious_feat)
    print("\nFeature Disentanglement:")
    for k, v in fds.items():
        print(f"  {k}: {v:.4f}")
    
    # Test tracker
    tracker = CausalMetricsTracker()
    tracker.update({'CFV': cfv['CFV_score'], 'ATE': 0.25})
    tracker.update({'CFV': 0.45, 'ATE': 0.30})
    print("\n" + tracker.summary())
    
    print("\n✓ All metric tests passed!")
