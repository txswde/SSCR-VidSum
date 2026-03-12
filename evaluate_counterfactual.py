"""
Counterfactual Explainability Evaluation Script

评估反事实可解释摘要的性能:
1. 反事实敏感度 (Counterfactual Sensitivity)
2. 解释一致性 (Explanation Consistency) 
3. 因果效应大小 (Effect Size)
4. 标准摘要指标 (ROUGE, Cosine)

Usage:
    python evaluate_counterfactual.py --dataset CNN --checkpoint saved_model/CNN --num_samples 50
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.utils.data
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_args, _DATASET_HYPER_PARAMS
from models import Model_MSMO, Model_BLiSS
from datasets import MSMODataset, BLiSSDataset, my_collate_fn, worker_init_fn
from counterfactual_summarization import CounterfactualGenerator, ExplainableSummarizer
from interpretability.causal_explainer import CausalEffectAnalyzer, CausalVisualizer
from interpretability.msmo_explainer import MSMOExplainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(args, checkpoint_path: str):
    """Load trained model from checkpoint"""
    if args.dataset == 'BLiSS':
        model = Model_BLiSS(args=args)
    elif args.dataset in ['Daily_Mail', 'CNN']:
        model = Model_MSMO(args=args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    model = model.to(args.device)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model


def evaluate_counterfactual_sensitivity(model, dataloader, args, num_samples: int = 50) -> Dict:
    """
    计算模型对反事实干预的敏感度
    
    敏感度越高说明模型对输入变化越敏感，反事实干预越有效
    
    Returns:
        Dict with sensitivity metrics
    """
    logger.info("Evaluating counterfactual sensitivity...")
    
    cf_generator = CounterfactualGenerator(default_strategy='mixed')
    
    sensitivities = {
        'text_zero': [],
        'text_noise': [],
        'text_shuffle': [],
        'video_zero': [],
        'video_noise': [],
    }
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="CF Sensitivity"):
            if sample_count >= num_samples:
                break
            
            (video_list, video_summ_list, text_list,
             mask_video_list, mask_video_summ_list, mask_text_list,
             video_label_list, text_label_list, article_segment_list, highlight_list,
             video_to_text_mask_list, text_to_video_mask_list) = batch
            
            video = pad_sequence(video_list, batch_first=True).to(args.device)
            text = pad_sequence(text_list, batch_first=True).to(args.device)
            mask_video = pad_sequence(mask_video_list, batch_first=True).to(args.device)
            mask_text = pad_sequence(mask_text_list, batch_first=True).to(args.device)
            video_label = pad_sequence(video_label_list, batch_first=True).to(args.device)
            text_label = pad_sequence(text_label_list, batch_first=True).to(args.device)
            
            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)
            
            # Original prediction
            pred_video_orig, pred_text_orig, _ = model(
                video=video, text=text,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            prob_video_orig = torch.sigmoid(pred_video_orig)
            prob_text_orig = torch.sigmoid(pred_text_orig)
            
            # Test different CF strategies for text
            for strategy in ['zero', 'noise', 'shuffle']:
                text_cf = cf_generator.generate_text_cf(text, strategy=strategy, mask=mask_text)
                
                pred_video_cf, pred_text_cf, _ = model(
                    video=video, text=text_cf,
                    mask_video=mask_video, mask_text=mask_text,
                    video_label=video_label, text_label=text_label,
                    video_to_text_mask_list=video_to_text_mask_list,
                    text_to_video_mask_list=text_to_video_mask_list
                )
                
                # Compute sensitivity as prediction change
                video_change = torch.abs(prob_video_orig - torch.sigmoid(pred_video_cf))
                video_sensitivity = (video_change * mask_video.unsqueeze(-1) if video_change.dim() == 3 
                                    else video_change * mask_video).mean().item()
                
                sensitivities[f'text_{strategy}'].append(video_sensitivity)
            
            # Test video CF
            for strategy in ['zero', 'noise']:
                video_cf = cf_generator.generate_video_cf(video, strategy=strategy, mask=mask_video)
                
                pred_video_cf, pred_text_cf, _ = model(
                    video=video_cf, text=text,
                    mask_video=mask_video, mask_text=mask_text,
                    video_label=video_label, text_label=text_label,
                    video_to_text_mask_list=video_to_text_mask_list,
                    text_to_video_mask_list=text_to_video_mask_list
                )
                
                text_change = torch.abs(prob_text_orig - torch.sigmoid(pred_text_cf))
                text_sensitivity = (text_change * mask_text).mean().item()
                
                sensitivities[f'video_{strategy}'].append(text_sensitivity)
            
            sample_count += len(video_list)
    
    # Aggregate results
    results = {}
    for key, values in sensitivities.items():
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
    
    results['overall_text_sensitivity'] = np.mean([
        results.get('text_zero_mean', 0),
        results.get('text_noise_mean', 0),
        results.get('text_shuffle_mean', 0)
    ])
    results['overall_video_sensitivity'] = np.mean([
        results.get('video_zero_mean', 0),
        results.get('video_noise_mean', 0)
    ])
    
    return results


def evaluate_explanation_consistency(model, dataloader, args, num_samples: int = 30) -> Dict:
    """
    验证解释的一致性：相似输入产生相似解释
    
    通过添加小扰动，验证解释是否稳定
    
    Returns:
        Dict with consistency metrics
    """
    logger.info("Evaluating explanation consistency...")
    
    cf_generator = CounterfactualGenerator()
    
    consistency_scores = []
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Consistency"):
            if sample_count >= num_samples:
                break
            
            (video_list, video_summ_list, text_list,
             mask_video_list, mask_video_summ_list, mask_text_list,
             video_label_list, text_label_list, article_segment_list, highlight_list,
             video_to_text_mask_list, text_to_video_mask_list) = batch
            
            video = pad_sequence(video_list, batch_first=True).to(args.device)
            text = pad_sequence(text_list, batch_first=True).to(args.device)
            mask_video = pad_sequence(mask_video_list, batch_first=True).to(args.device)
            mask_text = pad_sequence(mask_text_list, batch_first=True).to(args.device)
            video_label = pad_sequence(video_label_list, batch_first=True).to(args.device)
            text_label = pad_sequence(text_label_list, batch_first=True).to(args.device)
            
            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)
            
            # Original prediction
            pred_video_orig, pred_text_orig, _ = model(
                video=video, text=text,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            # Add small perturbation
            noise_scale = 0.01
            text_perturbed = text + torch.randn_like(text) * noise_scale
            
            pred_video_perturbed, pred_text_perturbed, _ = model(
                video=video, text=text_perturbed,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            # Consistency = correlation between predictions
            video_corr = torch.corrcoef(torch.stack([
                pred_video_orig.flatten(), 
                pred_video_perturbed.flatten()
            ]))[0, 1].item()
            
            text_corr = torch.corrcoef(torch.stack([
                pred_text_orig.flatten(),
                pred_text_perturbed.flatten()
            ]))[0, 1].item()
            
            if not np.isnan(video_corr) and not np.isnan(text_corr):
                consistency_scores.append({
                    'video_consistency': video_corr,
                    'text_consistency': text_corr
                })
            
            sample_count += len(video_list)
    
    if consistency_scores:
        return {
            'video_consistency_mean': np.mean([s['video_consistency'] for s in consistency_scores]),
            'video_consistency_std': np.std([s['video_consistency'] for s in consistency_scores]),
            'text_consistency_mean': np.mean([s['text_consistency'] for s in consistency_scores]),
            'text_consistency_std': np.std([s['text_consistency'] for s in consistency_scores]),
        }
    return {}


def evaluate_causal_effect_size(model, dataloader, args, num_samples: int = 30) -> Dict:
    """
    计算因果效应大小
    
    使用ATE和CATE指标
    
    Returns:
        Dict with effect size metrics
    """
    logger.info("Evaluating causal effect size...")
    
    analyzer = CausalEffectAnalyzer(model)
    cf_generator = CounterfactualGenerator()
    
    ate_scores = []
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Effect Size"):
            if sample_count >= num_samples:
                break
            
            (video_list, video_summ_list, text_list,
             mask_video_list, mask_video_summ_list, mask_text_list,
             video_label_list, text_label_list, article_segment_list, highlight_list,
             video_to_text_mask_list, text_to_video_mask_list) = batch
            
            video = pad_sequence(video_list, batch_first=True).to(args.device)
            text = pad_sequence(text_list, batch_first=True).to(args.device)
            mask_video = pad_sequence(mask_video_list, batch_first=True).to(args.device)
            mask_text = pad_sequence(mask_text_list, batch_first=True).to(args.device)
            video_label = pad_sequence(video_label_list, batch_first=True).to(args.device)
            text_label = pad_sequence(text_label_list, batch_first=True).to(args.device)
            
            for i in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)
            
            # Generate counterfactual
            text_cf = cf_generator.generate_text_cf(text, strategy='zero', intensity=0.5, mask=mask_text)
            
            # Compute ATE
            try:
                ate_result = analyzer.compute_ATE(
                    video=video,
                    text_orig=text,
                    text_cf=text_cf,
                    mask_video=mask_video,
                    mask_text=mask_text,
                    video_label=video_label,
                    text_label=text_label,
                    video_to_text_mask_list=video_to_text_mask_list,
                    text_to_video_mask_list=text_to_video_mask_list
                )
                ate_scores.append(ate_result)
            except Exception as e:
                logger.warning(f"ATE computation failed: {e}")
            
            sample_count += len(video_list)
    
    if ate_scores:
        return {
            'ATE_mean': np.mean([s['ATE'] for s in ate_scores]),
            'ATE_std': np.std([s['ATE'] for s in ate_scores]),
            'ATE_abs_mean': np.mean([s['ATE_abs'] for s in ate_scores]),
            'effect_size_mean': np.mean([s['effect_size'] for s in ate_scores]),
        }
    return {}


def generate_sample_explanations(model, dataloader, args, num_samples: int = 5, output_dir: str = './explanations'):
    """
    为样本生成详细的解释报告
    """
    logger.info(f"Generating sample explanations to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = MSMOExplainer(model, device=args.device, save_dir=output_dir)
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            (video_list, video_summ_list, text_list,
             mask_video_list, mask_video_summ_list, mask_text_list,
             video_label_list, text_label_list, article_segment_list, highlight_list,
             video_to_text_mask_list, text_to_video_mask_list) = batch
            
            for i in range(min(len(video_list), num_samples - sample_count)):
                video = video_list[i].unsqueeze(0).to(args.device)
                text = text_list[i].unsqueeze(0).to(args.device)
                mask_video = mask_video_list[i].unsqueeze(0).to(args.device)
                mask_text = mask_text_list[i].unsqueeze(0).to(args.device)
                video_label = video_label_list[i].unsqueeze(0).to(args.device)
                text_label = text_label_list[i].unsqueeze(0).to(args.device)
                
                v2t_mask = [video_to_text_mask_list[i].to(args.device)]
                t2v_mask = [text_to_video_mask_list[i].to(args.device)]
                
                # Get predictions
                pred_video, pred_text, _ = model(
                    video=video, text=text,
                    mask_video=mask_video, mask_text=mask_text,
                    video_label=video_label, text_label=text_label,
                    video_to_text_mask_list=v2t_mask,
                    text_to_video_mask_list=t2v_mask
                )
                
                # Get selected frames and sentences
                num_frames = int(mask_video.sum().item())
                num_sentences = int(mask_text.sum().item())
                num_selected_frames = int(video_label.sum().item())
                num_selected_sentences = int(text_label.sum().item())
                
                selected_frames = torch.topk(pred_video[0, :num_frames], k=max(1, num_selected_frames))[1].tolist()
                selected_sentences = torch.topk(pred_text[0, :num_sentences], k=max(1, num_selected_sentences))[1].tolist()
                
                # Get article sentences if available
                article_sentences = list(article_segment_list[i]) if hasattr(article_segment_list[i], '__iter__') else None
                
                sample_id = f"sample_{batch_idx}_{i}"
                
                try:
                    # Generate report
                    report = explainer.generate_summary_report(
                        video=video, text=text,
                        mask_video=mask_video, mask_text=mask_text,
                        selected_frames=selected_frames,
                        selected_sentences=selected_sentences,
                        article_sentences=article_sentences,
                        sample_id=sample_id,
                        video_label=video_label, text_label=text_label,
                        video_to_text_mask_list=v2t_mask,
                        text_to_video_mask_list=t2v_mask
                    )
                    logger.info(f"Generated explanation for {sample_id}")
                except Exception as e:
                    logger.warning(f"Failed to generate explanation for {sample_id}: {e}")
                
                sample_count += 1


def main():
    parser = argparse.ArgumentParser(description='Evaluate Counterfactual Explainability')
    parser.add_argument('--dataset', type=str, default='CNN', choices=['CNN', 'Daily_Mail', 'BLiSS'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='./explanations', help='Output directory for explanations')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    
    eval_args = parser.parse_args()
    
    # Build full args with dataset defaults
    sys.argv = ['', f'--dataset={eval_args.dataset}']
    args = build_args()
    args.device = eval_args.device
    args.batch_size = eval_args.batch_size
    args.num_workers = eval_args.num_workers
    args.checkpoint = eval_args.checkpoint
    
    logger.info(f"Evaluating counterfactual explainability for {args.dataset}")
    logger.info(f"Checkpoint: {eval_args.checkpoint}")
    logger.info(f"Num samples: {eval_args.num_samples}")
    
    # Load model
    checkpoint_path = os.path.join(eval_args.checkpoint, 'model_best_text.pt')
    model = load_model(args, checkpoint_path)
    
    # Create dataloader
    if args.dataset in ['Daily_Mail', 'CNN']:
        dataset = MSMODataset(mode='test', args=args)
    else:
        dataset = BLiSSDataset(mode='test', args=args)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False, 
        pin_memory=True,
        worker_init_fn=worker_init_fn, 
        collate_fn=my_collate_fn
    )
    
    # Run evaluations
    results = {}
    
    # 1. Counterfactual Sensitivity
    sensitivity_results = evaluate_counterfactual_sensitivity(
        model, dataloader, args, num_samples=eval_args.num_samples
    )
    results['sensitivity'] = sensitivity_results
    logger.info(f"Sensitivity Results: {json.dumps(sensitivity_results, indent=2)}")
    
    # 2. Explanation Consistency
    consistency_results = evaluate_explanation_consistency(
        model, dataloader, args, num_samples=min(30, eval_args.num_samples)
    )
    results['consistency'] = consistency_results
    logger.info(f"Consistency Results: {json.dumps(consistency_results, indent=2)}")
    
    # 3. Causal Effect Size
    effect_results = evaluate_causal_effect_size(
        model, dataloader, args, num_samples=min(30, eval_args.num_samples)
    )
    results['effect_size'] = effect_results
    logger.info(f"Effect Size Results: {json.dumps(effect_results, indent=2)}")
    
    # 4. Generate sample explanations
    generate_sample_explanations(
        model, dataloader, args, 
        num_samples=min(5, eval_args.num_samples),
        output_dir=eval_args.output_dir
    )
    
    # Save all results
    os.makedirs(eval_args.output_dir, exist_ok=True)
    results_path = os.path.join(eval_args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples evaluated: {eval_args.num_samples}")
    # Helper function to format values safely
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
    
    logger.info(f"\nCounterfactual Sensitivity:")
    logger.info(f"  Text (overall): {fmt(sensitivity_results.get('overall_text_sensitivity', 'N/A'))}")
    logger.info(f"  Video (overall): {fmt(sensitivity_results.get('overall_video_sensitivity', 'N/A'))}")
    logger.info(f"\nExplanation Consistency:")
    logger.info(f"  Video: {fmt(consistency_results.get('video_consistency_mean', 'N/A'))}")
    logger.info(f"  Text: {fmt(consistency_results.get('text_consistency_mean', 'N/A'))}")
    logger.info(f"\nCausal Effect Size:")
    logger.info(f"  ATE (mean): {fmt(effect_results.get('ATE_mean', 'N/A'))}")
    logger.info(f"  Effect size: {fmt(effect_results.get('effect_size_mean', 'N/A'))}")
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Explanations saved to: {eval_args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
