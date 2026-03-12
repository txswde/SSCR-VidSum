# TVSum 创新点泛化评估报告

*Generated: 2026-02-23 10:55:16*

## 评估方法说明

- **Rank Correlation** 在 **segment-level** 计算（不做逐帧上采样），避免 ties 造成的失真
- 同时报告 **macro-average**（每个视频等权）和 **weighted-average**（按 segment 数量加权）
- 对两种 attention mask 进行 sanity check：uniform（线性分桶）和 all-ones（全连接）

## 2×2 消融结果

### F1-Score (5-Fold Cross Validation)

| | PSAL OFF | PSAL ON |
|---|---|---|
| **DCR OFF** | **0.6191** | **0.6274** |
| **DCR ON** | **0.6241** | **0.6219** |

### Rank Correlation — Uniform Mask

| Model | ρ (macro) | τ (macro) | ρ (weighted) | τ (weighted) | Valid/Total |
|---|---|---|---|---|---|
| Baseline | 0.2667 | 0.1848 | 0.3048 | 0.2108 | 50/50 |
| DCR Only | 0.2057 | 0.1408 | 0.2323 | 0.1584 | 50/50 |
| PSAL Only | 0.3006 | 0.2088 | 0.3463 | 0.2397 | 50/50 |
| Full | 0.2305 | 0.1575 | 0.2550 | 0.1749 | 50/50 |

### Rank Correlation — All-Ones Mask

| Model | ρ (macro) | τ (macro) | ρ (weighted) | τ (weighted) | Valid/Total |
|---|---|---|---|---|---|
| Baseline | 0.3024 | 0.2100 | 0.3515 | 0.2439 | 50/50 |
| DCR Only | 0.2168 | 0.1491 | 0.2193 | 0.1502 | 50/50 |
| PSAL Only | 0.3163 | 0.2211 | 0.3651 | 0.2542 | 50/50 |
| Full | 0.2588 | 0.1786 | 0.2876 | 0.1987 | 50/50 |

## 逐 Split 详情

### Baseline

| Split | F1-Score |
|---|---|
| split0 | 0.6197 |
| split1 | 0.5856 |
| split2 | 0.6383 |
| split3 | 0.6331 |
| split4 | 0.6188 |

### DCR Only

| Split | F1-Score |
|---|---|
| split0 | 0.6208 |
| split1 | 0.6053 |
| split2 | 0.6379 |
| split3 | 0.6222 |
| split4 | 0.6341 |

### PSAL Only

| Split | F1-Score |
|---|---|
| split0 | 0.6367 |
| split1 | 0.5953 |
| split2 | 0.6404 |
| split3 | 0.6226 |
| split4 | 0.6422 |

### Full

| Split | F1-Score |
|---|---|
| split0 | 0.6188 |
| split1 | 0.5871 |
| split2 | 0.6363 |
| split3 | 0.6297 |
| split4 | 0.6375 |

## 可视化

### 消融柱状图

![消融柱状图](D:\ai-agent\A2Summ-main\logs\analysis\tvsum_ablation\tvsum_ablation_bar_chart.png)

### 训练曲线

![训练曲线](D:\ai-agent\A2Summ-main\logs\analysis\tvsum_ablation\tvsum_training_curves.png)

### Δ 改进热力图

![Δ 改进热力图](D:\ai-agent\A2Summ-main\logs\analysis\tvsum_ablation\tvsum_delta_heatmap.png)

### Mask Sanity Check

![Mask Sanity Check](D:\ai-agent\A2Summ-main\logs\analysis\tvsum_ablation\tvsum_mask_sanity_check.png)

## 分析

- **DCR Only** vs Baseline: F1 ↑ 0.81% (0.6191 → 0.6241)
- **PSAL Only** vs Baseline: F1 ↑ 1.34% (0.6191 → 0.6274)
- **Full** vs Baseline: F1 ↑ 0.45% (0.6191 → 0.6219)
