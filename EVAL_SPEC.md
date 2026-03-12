# Evaluation Specification — BLiSS Alignment

> Canonical reference for all evaluation decisions.  
> Script: `evaluate_bliss_alignment_improved.py` v2.0

## 1. Causal Matrix Construction

| Step | Detail |
|------|--------|
| Intervention | Zero-out one sentence embedding at a time |
| Effect | `|σ(pred_orig) − σ(pred_masked)|` per frame |
| Output shape | `(num_frames, num_sentences)` |
| Normalization | **Per-sample global-max** → [0, 1] |
| Trimming | Matrix cropped to GT shape: `causal[:F, :S]` |

## 2. Ground-Truth (GT) Binarization

```
gt_binary = (gt_alignment > 0).astype(float)
```

- `> 0` is **strict**: zero cells are treated as "no alignment".
- BLiSS GT sparsity is ~15–30% non-zero.

## 3. Metrics

### 3.1 Primary Metrics (Paper Body)

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **AUC-PR** | None (threshold-free) | Area under Precision-Recall curve via `sklearn` |
| **F1@fixed-rule** | 95th percentile of causal values | Adaptive but rule-based, not oracle |
| **F1@0.05** | 0.05 (hard-coded) | Lowest preset threshold |
| **Top-K Hit Rate** | K = #GT positives | `|top-K ∩ GT| / K` |
| **Causal Sensitivity** | None | Spearman ρ(sentence_effects, GT frame counts) |

### 3.2 Appendix Metrics (Oracle)

| Metric | Note |
|--------|------|
| Best-F1 | Max F1 across preset thresholds [0.05 … 0.7] — oracle |
| Best-IoU | IoU at the Best-F1 threshold — oracle |

### 3.3 Legacy (Not Recommended)

Pearson GTC, Spearman GTC, Energy-in-mask — kept for backward compatibility only.

## 4. Averaging

**Macro (per-sample → mean)**:
1. Compute each metric per sample.
2. Report `mean`, `std`, `median` across all valid samples.

This is NOT micro-averaging (pooling all TP/FP/FN globally).

## 5. Threshold Binarization

```
pred_binary = (causal_matrix >= threshold)
```

- **Inclusive** (`>=`): a value exactly equal to the threshold is considered positive.

## 6. Statistical Stability

| Item | Value |
|------|-------|
| Global seed | 42 (numpy + torch + CUDNN deterministic + python random) |
| Bootstrap | 5000 resamples, percentile method |
| CI | 95% (2.5th–97.5th percentile of bootstrap distribution) |
| CI metrics | All 7 key metrics (5 primary + 2 appendix) |

## 7. Reproducibility

- JSON output includes `_metadata` block with: seed, script version, timestamp, torch/numpy versions, git commit.
- CUDNN: `deterministic=True`, `benchmark=False`.
- Test samples iterated in index order (0, 1, 2, …, N-1), no shuffle.
