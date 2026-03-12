"""
PSAL Attention Visualization Script
====================================
Compare cross-modal attention patterns between:
  - Baseline (with GT temporal alignment mask)
  - PSAL (prior-free semantic alignment, full attention)

Generates:
  1. Heatmap comparison: GT mask vs Baseline attention vs PSAL attention
  2. Layer-wise attention evolution (6 layers)
  3. Attention entropy statistics (quantifying attention concentration)

Usage:
  python visualize_attention.py
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns

# ============================================================
# 1. Attention Hook — non-invasively extract attention weights
# ============================================================

class AttentionHook:
    """Register forward hooks on MultiHeadAttention modules to capture
    the softmax attention weights without modifying models.py."""

    def __init__(self):
        self.attention_maps = []   # per-layer attention maps
        self._hooks = []

    def _hook_fn(self, module, input_args, output):
        """Re-compute attention weights inside the hook.
        
        We replicate the lightweight Q·K^T / sqrt(d) + mask → softmax
        computation so that the stored map is *before* dropout.
        """
        # input_args: (q_input, k_input, v_input) + mask via kwargs
        # We need to recompute because the original forward doesn't expose att.
        q_in = input_args[0]  # q input to the linear layer
        k_in = input_args[1] if len(input_args) > 1 and torch.is_tensor(input_args[1]) else q_in
        
        # Get mask from kwargs or from the module's forward call context
        # Since hooks don't easily get kwargs, we store mask via a wrapper
        mask = getattr(module, '_current_mask', None)
        
        with torch.no_grad():
            q = module.q(q_in).transpose(0, 1).contiguous()
            k = module.k(k_in).transpose(0, 1).contiguous()
            
            b = q.size(1) * module._heads
            head_dims = module._head_dims
            
            q = q.view(-1, b, head_dims).transpose(0, 1)
            k = k.view(-1, b, head_dims).transpose(0, 1)
            
            att = torch.bmm(q, k.transpose(1, 2)) / head_dims**0.5
            
            if mask is not None:
                mask_transformed = torch.where(mask > 0, 0.0, float('-inf'))
                mask_transformed = mask_transformed.repeat_interleave(module._heads, dim=0)
                att = att + mask_transformed
            
            att = att.softmax(-1)  # [B*heads, seq_len, seq_len]
            
            # Average over heads: reshape to [B, heads, S, S] then mean over heads
            B_actual = att.size(0) // module._heads
            att_heads = att.view(B_actual, module._heads, att.size(1), att.size(2))
            att_avg = att_heads.mean(dim=1)  # [B, S, S]
            
            self.attention_maps.append(att_avg.cpu())

    def register(self, model):
        """Register hooks on all MultiWayTransformer attention fusion layers."""
        for name, module in model.named_modules():
            if hasattr(module, 'attn_fusion'):
                # We hook the attn_fusion (MultiHeadAttention) inside each MultiWayTransformer
                hook = module.attn_fusion.register_forward_hook(self._hook_fn)
                self._hooks.append(hook)
        return self

    def clear(self):
        self.attention_maps = []

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def patch_forward_for_mask_capture(model):
    """Monkey-patch MultiWayTransformer.forward to store mask on attn_fusion
    so the hook can access it."""
    from models import MultiWayTransformer
    
    _orig_forward = MultiWayTransformer.forward

    def _patched_forward(self, fused, mask_fused, N_video, N_text):
        # Store the mask on the attention module so the hook can read it
        self.attn_fusion._current_mask = mask_fused
        return _orig_forward(self, fused, mask_fused, N_video, N_text)

    MultiWayTransformer.forward = _patched_forward


# ============================================================
# 2. Model & Data Loading
# ============================================================

def load_model(checkpoint_dir, args_override=None):
    """Load a trained Model_BLiSS from checkpoint."""
    import yaml
    from models import Model_BLiSS

    # Load saved args
    args_path = os.path.join(checkpoint_dir, 'args.yml')
    with open(args_path, 'r') as f:
        saved_args = yaml.safe_load(f)

    # Build a namespace from saved args
    ns = argparse.Namespace(**saved_args)
    if args_override:
        for k, v in args_override.items():
            setattr(ns, k, v)

    model = Model_BLiSS(args=ns)
    
    # Try loading best text checkpoint
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint', 'model_best_text.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'checkpoint', 'model_best_video.pt')
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    return model, ns


def load_test_samples(data_root='data', dataset='BLiSS', num_samples=3, seed=42):
    """Load a few test samples from BLiSS dataset."""
    gt = json.load(open(f'{data_root}/{dataset}/annotation/test.json'))
    clip_ids = list(gt.keys())
    
    rng = np.random.RandomState(seed)
    selected_ids = rng.choice(clip_ids, size=min(num_samples, len(clip_ids)), replace=False)
    
    video_dict = np.load(f'{data_root}/{dataset}/feature/video_clip_test.npy', allow_pickle=True).item()
    text_dict = np.load(f'{data_root}/{dataset}/feature/text_roberta_test.npy', allow_pickle=True).item()
    
    samples = []
    for clip_id in selected_ids:
        video = torch.tensor(video_dict[clip_id], dtype=torch.float32)
        text = torch.tensor(text_dict[clip_id], dtype=torch.float32)
        
        video_label = torch.tensor(gt[clip_id]['video_label'], dtype=torch.long)
        text_label = torch.tensor(gt[clip_id]['text_label'], dtype=torch.long)
        
        num_frame = gt[clip_id]['num_frame']
        num_sentence = gt[clip_id]['num_sentence']
        time_index = gt[clip_id]['sentence_time']
        sentences = gt[clip_id].get('sentence', [f'sent_{j}' for j in range(num_sentence)])
        
        # Build GT alignment mask
        video_to_text_mask = torch.zeros((num_frame, num_sentence), dtype=torch.long)
        text_to_video_mask = torch.zeros((num_sentence, num_frame), dtype=torch.long)
        for j in range(num_sentence):
            start_frame, end_frame = time_index[j]
            video_to_text_mask[start_frame:end_frame, j] = 1
            text_to_video_mask[j, start_frame:end_frame] = 1
        
        samples.append({
            'clip_id': clip_id,
            'video': video,
            'text': text,
            'video_label': video_label,
            'text_label': text_label,
            'num_frame': num_frame,
            'num_sentence': num_sentence,
            'sentences': sentences,
            'video_to_text_mask_gt': video_to_text_mask,
            'text_to_video_mask_gt': text_to_video_mask,
        })
    
    return samples


# ============================================================
# 3. Run Inference & Extract Attention
# ============================================================

def run_inference_with_hooks(model, sample, disable_alignment_mask=False, device='cpu'):
    """Run single-sample inference and capture per-layer attention maps."""
    hook = AttentionHook()
    hook.register(model)
    
    model = model.to(device)
    
    video = sample['video'].unsqueeze(0).to(device)
    text = sample['text'].unsqueeze(0).to(device)
    mask_video = torch.ones(1, sample['num_frame'], dtype=torch.long).to(device)
    mask_text = torch.ones(1, sample['num_sentence'], dtype=torch.long).to(device)
    video_label = sample['video_label'].unsqueeze(0).to(device)
    text_label = sample['text_label'].unsqueeze(0).to(device)
    
    if disable_alignment_mask:
        v2t_mask = torch.ones_like(sample['video_to_text_mask_gt'])
        t2v_mask = torch.ones_like(sample['text_to_video_mask_gt'])
    else:
        v2t_mask = sample['video_to_text_mask_gt'].clone()
        t2v_mask = sample['text_to_video_mask_gt'].clone()
    
    v2t_mask = v2t_mask.to(device)
    t2v_mask = t2v_mask.to(device)
    
    with torch.no_grad():
        model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=[v2t_mask],
            text_to_video_mask_list=[t2v_mask],
        )
    
    attention_maps = [a[0] for a in hook.attention_maps]  # remove batch dim
    hook.remove()
    
    return attention_maps


def extract_cross_modal_attention(attn_map, N_video, N_text):
    """Extract video→text and text→video blocks from the fused attention map.
    
    attn_map: [N_video+N_text, N_video+N_text] (includes CLS tokens)
    Returns: video_to_text [N_video-1, N_text-1] (excluding CLS tokens)
    """
    # video→text block: rows [1:N_video], cols [N_video+1:]
    v2t = attn_map[1:N_video, N_video+1:N_video+N_text].numpy()
    # text→video block: rows [N_video+1:], cols [1:N_video]
    t2v = attn_map[N_video+1:N_video+N_text, 1:N_video].numpy()
    return v2t, t2v


# ============================================================
# 4. Visualization Functions
# ============================================================

def plot_attention_comparison(sample, baseline_attns, psal_attns, output_dir, sample_idx=0):
    """Generate 3-column comparison: GT mask | Baseline attention | PSAL attention.
    
    Shows the last layer's cross-modal attention (video→text).
    """
    N_v = sample['num_frame'] + 1  # +1 for CLS
    N_t = sample['num_sentence'] + 1
    
    gt_mask = sample['video_to_text_mask_gt'].numpy()  # [num_frame, num_sentence]
    
    # Last layer attention
    baseline_v2t, _ = extract_cross_modal_attention(baseline_attns[-1], N_v, N_t)
    psal_v2t, _ = extract_cross_modal_attention(psal_attns[-1], N_v, N_t)
    
    # Truncate to valid region
    nf, ns = gt_mask.shape
    baseline_v2t = baseline_v2t[:nf, :ns]
    psal_v2t = psal_v2t[:nf, :ns]
    
    # Normalize for better visualization
    baseline_v2t_norm = baseline_v2t / (baseline_v2t.max() + 1e-8)
    psal_v2t_norm = psal_v2t / (psal_v2t.max() + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Cross-Modal Attention Comparison (Sample: {sample["clip_id"]})', 
                 fontsize=14, fontweight='bold')
    
    # GT alignment mask
    im0 = axes[0].imshow(gt_mask, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[0].set_title('(a) GT Temporal\nAlignment Mask', fontsize=12)
    axes[0].set_xlabel('Text Sentence Index')
    axes[0].set_ylabel('Video Frame Index')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Baseline learned attention (with GT mask constraint)
    im1 = axes[1].imshow(baseline_v2t_norm, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[1].set_title('(b) Baseline Attention\n(with GT Mask)', fontsize=12)
    axes[1].set_xlabel('Text Sentence Index')
    axes[1].set_ylabel('Video Frame Index')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # PSAL learned attention (no mask constraint)
    im2 = axes[2].imshow(psal_v2t_norm, aspect='auto', cmap='Greens', interpolation='nearest')
    axes[2].set_title('(c) PSAL Attention\n(Prior-free)', fontsize=12)
    axes[2].set_xlabel('Text Sentence Index')
    axes[2].set_ylabel('Video Frame Index')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'attention_comparison_sample{sample_idx}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_evolution(sample, psal_attns, output_dir, sample_idx=0):
    """Show how PSAL attention evolves across 6 transformer layers."""
    N_v = sample['num_frame'] + 1
    N_t = sample['num_sentence'] + 1
    nf = sample['num_frame']
    ns = sample['num_sentence']
    
    num_layers = len(psal_attns)
    cols = min(num_layers, 6)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f'PSAL Attention Evolution Across Layers (Sample: {sample["clip_id"]})', 
                 fontsize=14, fontweight='bold')
    
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    
    for layer_idx in range(num_layers):
        v2t, _ = extract_cross_modal_attention(psal_attns[layer_idx], N_v, N_t)
        v2t = v2t[:nf, :ns]
        v2t_norm = v2t / (v2t.max() + 1e-8)
        
        ax = axes[layer_idx]
        im = ax.imshow(v2t_norm, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=11)
        ax.set_xlabel('Text Sent.')
        ax.set_ylabel('Video Frame')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused axes
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'psal_layer_evolution_sample{sample_idx}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_entropy_comparison(all_baseline_entropies, all_psal_entropies, output_dir):
    """Compare attention entropy between Baseline and PSAL across layers.
    
    Lower entropy = more concentrated/focused attention.
    """
    num_layers = len(all_baseline_entropies)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Attention Entropy Analysis: Baseline vs PSAL', fontsize=14, fontweight='bold')
    
    # Line plot: mean entropy per layer
    layers = list(range(1, num_layers + 1))
    baseline_means = [np.mean(e) for e in all_baseline_entropies]
    psal_means = [np.mean(e) for e in all_psal_entropies]
    baseline_stds = [np.std(e) for e in all_baseline_entropies]
    psal_stds = [np.std(e) for e in all_psal_entropies]
    
    ax1.errorbar(layers, baseline_means, yerr=baseline_stds, marker='o', 
                 label='Baseline (GT Mask)', color='#e74c3c', capsize=4, linewidth=2)
    ax1.errorbar(layers, psal_means, yerr=psal_stds, marker='s', 
                 label='PSAL (Prior-free)', color='#27ae60', capsize=4, linewidth=2)
    ax1.set_xlabel('Transformer Layer', fontsize=12)
    ax1.set_ylabel('Mean Cross-Modal Attention Entropy', fontsize=12)
    ax1.set_title('(a) Entropy per Layer', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.3)
    
    # Bar plot: overall entropy distribution
    baseline_all = np.concatenate(all_baseline_entropies)
    psal_all = np.concatenate(all_psal_entropies)
    
    ax2.hist(baseline_all, bins=30, alpha=0.6, label='Baseline', color='#e74c3c', density=True)
    ax2.hist(psal_all, bins=30, alpha=0.6, label='PSAL', color='#27ae60', density=True)
    ax2.set_xlabel('Attention Entropy (nats)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('(b) Entropy Distribution\n(All Layers, All Samples)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'attention_entropy_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_attention_entropy(attn_map, N_v, N_t, num_frame, num_sentence):
    """Compute entropy of video→text cross-modal attention for each frame."""
    v2t, _ = extract_cross_modal_attention(attn_map, N_v, N_t)
    v2t = v2t[:num_frame, :num_sentence]
    
    # Normalize each row to a probability distribution
    row_sums = v2t.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    v2t_prob = v2t / row_sums
    
    # Compute entropy: H = -sum(p * log(p))
    v2t_prob = np.clip(v2t_prob, 1e-10, 1.0)
    entropy = -np.sum(v2t_prob * np.log(v2t_prob), axis=1)  # [num_frame]
    
    return entropy


def plot_alignment_quality(samples, all_psal_attns, output_dir):
    """Quantitatively evaluate PSAL's learned alignment vs GT."""
    results = []
    
    for s_idx, (sample, psal_attns) in enumerate(zip(samples, all_psal_attns)):
        N_v = sample['num_frame'] + 1
        N_t = sample['num_sentence'] + 1
        nf = sample['num_frame']
        ns = sample['num_sentence']
        gt_mask = sample['video_to_text_mask_gt'].numpy()
        
        # Use last layer attention
        v2t, _ = extract_cross_modal_attention(psal_attns[-1], N_v, N_t)
        v2t = v2t[:nf, :ns]
        
        # For each frame, find the sentence with highest attention
        pred_align = np.argmax(v2t, axis=1)
        gt_align = np.argmax(gt_mask, axis=1) if gt_mask.sum() > 0 else np.zeros(nf)
        
        # Compute alignment accuracy (frame → most attended sentence matches GT)
        gt_valid = gt_mask.sum(axis=1) > 0
        if gt_valid.sum() > 0:
            accuracy = (pred_align[gt_valid] == gt_align[gt_valid]).mean()
        else:
            accuracy = 0.0
        
        # Compute soft IoU: overlap between thresholded attention and GT mask
        v2t_binary = (v2t > np.percentile(v2t, 70)).astype(float)
        intersection = np.sum(v2t_binary * gt_mask)
        union = np.sum(np.maximum(v2t_binary, gt_mask))
        iou = intersection / (union + 1e-8)
        
        results.append({'clip_id': sample['clip_id'], 'accuracy': accuracy, 'iou': iou})
    
    # Summary table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    cell_text = [[r['clip_id'][:20], f"{r['accuracy']:.3f}", f"{r['iou']:.3f}"] for r in results]
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    cell_text.append(['**Average**', f'{avg_acc:.3f}', f'{avg_iou:.3f}'])
    
    table = ax.table(
        cellText=cell_text,
        colLabels=['Sample ID', 'Alignment Acc.', 'Soft IoU'],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header
    for j in range(3):
        table[0, j].set_facecolor('#3498db')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Color the average row
    for j in range(3):
        table[len(cell_text), j].set_facecolor('#ecf0f1')
        table[len(cell_text), j].set_text_props(fontweight='bold')
    
    ax.set_title('PSAL Alignment Quality vs GT\n(Higher = model discovers correct alignment)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    save_path = os.path.join(output_dir, 'psal_alignment_quality.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    return results


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PSAL Attention Visualization')
    parser.add_argument('--baseline_dir', type=str, 
                        default='logs/BLiss/BLiSS_pure_baseline',
                        help='Checkpoint directory for baseline model')
    parser.add_argument('--psal_dir', type=str, 
                        default='logs/BLiss/BLiSS_unsupervised',
                        help='Checkpoint directory for PSAL model')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of test samples to visualize')
    parser.add_argument('--output_dir', type=str, 
                        default='output/attention_visualization')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    viz_args = parser.parse_args()
    
    os.makedirs(viz_args.output_dir, exist_ok=True)
    
    # Patch forward to capture mask
    patch_forward_for_mask_capture(None)
    
    print("=" * 60)
    print("PSAL Attention Visualization")
    print("=" * 60)
    
    # Load models
    print("\n[1/4] Loading models...")
    baseline_model, baseline_args = load_model(viz_args.baseline_dir)
    psal_model, psal_args = load_model(viz_args.psal_dir)
    
    baseline_model = baseline_model.to(viz_args.device)
    psal_model = psal_model.to(viz_args.device)
    
    # Load test samples
    print(f"\n[2/4] Loading {viz_args.num_samples} test samples...")
    samples = load_test_samples(
        data_root=viz_args.data_root, 
        dataset='BLiSS', 
        num_samples=viz_args.num_samples
    )
    print(f"  Loaded {len(samples)} samples")
    
    # Run inference and collect attention maps
    print("\n[3/4] Extracting attention maps...")
    all_baseline_attns = []
    all_psal_attns = []
    all_baseline_entropies = [[] for _ in range(baseline_args.num_layers)]
    all_psal_entropies = [[] for _ in range(psal_args.num_layers)]
    
    for s_idx, sample in enumerate(samples):
        print(f"  Sample {s_idx + 1}/{len(samples)}: {sample['clip_id']} "
              f"({sample['num_frame']} frames, {sample['num_sentence']} sentences)")
        
        # Baseline: use GT alignment mask
        baseline_attns = run_inference_with_hooks(
            baseline_model, sample, disable_alignment_mask=False, device=viz_args.device
        )
        all_baseline_attns.append(baseline_attns)
        
        # PSAL: use full attention (no mask)
        psal_attns = run_inference_with_hooks(
            psal_model, sample, disable_alignment_mask=True, device=viz_args.device
        )
        all_psal_attns.append(psal_attns)
        
        # Compute entropy per layer
        N_v = sample['num_frame'] + 1
        N_t = sample['num_sentence'] + 1
        for layer_idx in range(len(baseline_attns)):
            be = compute_attention_entropy(baseline_attns[layer_idx], N_v, N_t, 
                                           sample['num_frame'], sample['num_sentence'])
            all_baseline_entropies[layer_idx].append(be)
            
            pe = compute_attention_entropy(psal_attns[layer_idx], N_v, N_t, 
                                           sample['num_frame'], sample['num_sentence'])
            all_psal_entropies[layer_idx].append(pe)
    
    # Concatenate entropies
    all_baseline_entropies = [np.concatenate(e) for e in all_baseline_entropies]
    all_psal_entropies = [np.concatenate(e) for e in all_psal_entropies]
    
    # Generate visualizations
    print(f"\n[4/4] Generating visualizations → {viz_args.output_dir}/")
    
    for s_idx, sample in enumerate(samples):
        # Fig 1: 3-column comparison (GT mask vs Baseline vs PSAL)
        plot_attention_comparison(
            sample, all_baseline_attns[s_idx], all_psal_attns[s_idx], 
            viz_args.output_dir, sample_idx=s_idx
        )
        
        # Fig 2: Layer-wise evolution for PSAL
        plot_layer_evolution(
            sample, all_psal_attns[s_idx], 
            viz_args.output_dir, sample_idx=s_idx
        )
    
    # Fig 3: Entropy comparison
    plot_entropy_comparison(
        all_baseline_entropies, all_psal_entropies, viz_args.output_dir
    )
    
    # Fig 4: Alignment quality table
    results = plot_alignment_quality(samples, all_psal_attns, viz_args.output_dir)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output directory: {viz_args.output_dir}")
    print(f"Generated files:")
    for f in sorted(os.listdir(viz_args.output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")
    print("=" * 60)
    
    # Print alignment summary
    print("\nPSAL Alignment Quality Summary:")
    for r in results:
        print(f"  {r['clip_id'][:25]:25s} Acc: {r['accuracy']:.3f}  IoU: {r['iou']:.3f}")
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    print(f"  {'Average':25s} Acc: {avg_acc:.3f}  IoU: {avg_iou:.3f}")


if __name__ == '__main__':
    main()
