# BLiSS 实验参数对比分析

本文档对比了四个 BLiSS 实验的配置差异。

## 实验列表

1. **BLiSS_pure_baseline**: 纯基线模型，无因果干预。
2. **BLiSS_unsupervised**: 无监督设置（可能缺省部分因果参数）。
3. **BLiSS_unsupervised_yinguo**: 引入因果对比 (Causal Contrast) 和差异因果 (Diff Causal)，但未启用因果对齐 (Causal Alignment)。
4. **BLiSS_unsupervised_yinguo_duiqi_corrected**: 全功能模型，启用了因果对齐。

## 参数对比表

| 参数 | pure_baseline | unsupervised | unsupervised_yinguo | unsupervised_yinguo_duiqi_corrected |
|:---|:---:|:---:|:---:|:---:|
| **max_epoch** | 20 | 50 | 20 | 25 |
| **enable_causal_alignment** | `false` | (Default `false`)* | `false` | `true` |
| **enable_diff_causal** | `false` | (Default `false`)* | `true` | `true` |
| **lambda_causal_contrast** | 0.0 | 0.3 | 0.3 | 0.3 |
| **lambda_causal_alignment** | 0.5 (unused) | (Default) | 0.5 (unused) | 0.5 |
| **lambda_diff_causal** | 0.0 | 0.0 (implied) | 0.3 | 0.3 |
| **disable_alignment_mask** | `false` | `true` | `true` | `true` |

> *注：`unsupervised` 的配置文件中未显式包含 `enable_causal_alignment` 和 `enable_diff_causal`，根据代码逻辑通常默认为 False。

## 关键发现

1. **基线 (Baseline)**: 所有因果损失权重均为 0.0，且显式禁用了相关功能。
2. **因果引入 (Yinguo)**: `unsupervised_yinguo` 引入了 `lambda_causal_contrast` (0.3) 和 `lambda_diff_causal` (0.3)，但 **未启用** `enable_causal_alignment`。
3. **全对齐 (Duiqi Corrected)**: 这是最完整的配置，不仅包含上述因果损失，还显式启用了 `enable_causal_alignment: true`，这与 `causal_alignment_discovery.py` 的功能对应。
4. **Mask 设置**: 后三个实验均设置了 `disable_alignment_mask: true`，这可能意味着在这些实验中不使用传统的对齐 Mask，而是依赖因果对齐发现或无监督机制。

## 结论

`BLiSS_unsupervised_yinguo_duiqi_corrected` 是验证 "Innovation 2: Intervention Dissimilarity (Causal Alignment Discovery)" 的关键实验。如果需要验证因果对齐的有效性，应重点关注此实验与 `BLiSS_unsupervised_yinguo`（有因果无对齐）的对比。
