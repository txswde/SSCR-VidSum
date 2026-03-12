# A2Summ 模块说明文档

本文档说明项目中各模块的用途和适用数据集。

---

## 核心模块

### 1. 模型定义
| 文件 | 主要类 | 用途 | 适用数据集 |
|:-----|:-------|:-----|:-----------|
| `models.py` | `Model_BLiSS` | BLiSS 数据集专用模型 | BLiSS |
| `models.py` | `Model_MSMO` | CNN/Daily_Mail 数据集专用模型 | CNN, Daily_Mail |
| `models.py` | `Model_VideoSumm` | SumMe/TVSum 数据集专用模型 | SumMe, TVSum |

### 2. 损失函数
| 文件 | 主要类 | 用途 | 适用数据集 |
|:-----|:-------|:-----|:-----------|
| `losses.py` | `Dual_Contrastive_Loss` | 双对比损失（原始A2Summ） | 全部 |
| `causal_contrastive_loss.py` | `CausalContrastiveLoss` | 因果对比损失 | **BLiSS** |
| `causal_contrastive_loss.py` | `CausalEffectLoss` | 因果效应损失 | **BLiSS** |
| `causal_alignment_discovery.py` | `CausalAlignmentDiscovery` | ⭐ 干预相异度对齐发现 | **BLiSS** |
| `counterfactual_summarization.py` | `CounterfactualSummarizationLoss` | 反事实摘要损失 | CNN, Daily_Mail |

### 3. 数据集
| 文件 | 主要类 | 用途 |
|:-----|:-------|:-----|
| `datasets.py` | `BLiSSDataset` | BLiSS 数据加载 |
| `datasets.py` | `MSMODataset` | CNN/Daily_Mail 数据加载 |
| `datasets.py` | `VideoSummDataset` | SumMe/TVSum 数据加载 |

### 4. 训练脚本
| 文件 | 用途 | 适用数据集 |
|:-----|:-----|:-----------|
| `train.py` | 统一入口 | 全部 |
| `train_msmo.py` | BLiSS/CNN/Daily_Mail 训练 | BLiSS, CNN, Daily_Mail |
| `train_videosumm.py` | SumMe/TVSum 训练 | SumMe, TVSum |

---

## 未使用模块 (unused/)

以下模块已移至 `unused/` 目录，它们在 **BLiSS 训练流程中不被使用**：

| 文件 | 原因 |
|:-----|:-----|
| `unused/temporal_alignment_loss.py` | 设计用于 SumMe/TVSum，但实际上 BLiSS 使用 `CausalAlignmentDiscovery` |
| `unused/text_necessity_loss.py` | 设计用于 SumMe/TVSum，BLiSS 训练中未调用 |

> **注意**：这些模块仍可被 `causal_contrastive_loss.py` 中的 `CombinedCausalLoss` 类导入使用（用于 SumMe/TVSum）。

---

## BLiSS 数据集创新点

针对 BLiSS 数据集，项目实现了两个核心创新：

### 创新 1: Causal Contrastive Loss (因果对比损失)

**文件**: `causal_contrastive_loss.py`  
**类**: `CausalContrastiveLoss`, `CausalEffectLoss`

**核心思想**：
- 轻微干预（mask 1个句子）→ 特征应保持不变
- 剧烈干预（mask 30-50%句子）→ 特征应产生变化

**启用方式**：
```bash
python train.py --dataset BLiSS --lambda_causal_contrast 0.3
```

### 创新 2: Intervention Dissimilarity (干预相异度) ⭐

**文件**: `causal_alignment_discovery.py`  
**类**: `CausalAlignmentDiscovery`

**核心思想**：
```
Alignment_{i,j} = |P(Y_i|V, T) - P(Y_i|V, T_{-j})|
```

通过逐一 mask 文本句子并观察视频预测变化来发现跨模态对齐，无需 GT 对齐标签。

**启用方式**：
```bash
python train.py --dataset BLiSS \
    --disable_alignment_mask \
    --enable_causal_alignment \
    --lambda_causal_alignment 0.5
```

---

## 配置文件

详细的实验配置见 `configs/bliss_experiments.yaml`。

---

*文档更新时间: 2026-02-02*
