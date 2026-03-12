"""
涌现对齐实验脚本 (Emergent Alignment Experiment)
自动运行对比实验：有监督对齐 vs 无监督对齐

Usage:
    python run_alignment_experiment.py
"""

import os
import subprocess
import sys
import json
from datetime import datetime


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"命令: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode


def main():
    print("\n" + "="*70)
    print("   涌现对齐实验 (Emergent Alignment Experiment)")
    print("   验证模型能否不使用时间戳监督学习跨模态对齐")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./experiments/emergent_alignment_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    experiments = [
        {
            "name": "baseline_supervised",
            "description": "有监督基线: 使用GT时间戳对齐mask",
            "cmd": "python train.py --dataset BLiSS --max_epoch 30 --batch_size 64 --print_freq 50 --eval_freq 5",
            "model_dir": "logs/BLiSS"
        },
        {
            "name": "unsupervised",
            "description": "无监督线: 禁用对齐mask (全1矩阵)",
            "cmd": "python train.py --dataset BLiSS --max_epoch 30 --batch_size 64 --print_freq 50 --eval_freq 5 --disable_alignment_mask",
            "model_dir": "logs/BLiSS"
        },
        {
            "name": "unsupervised_causal",
            "description": "无监督因果对齐: 禁用对齐mask + 因果对比损失",
            "cmd": "python train.py --dataset BLiSS --max_epoch 30 --batch_size 64 --print_freq 50 --eval_freq 5 --disable_alignment_mask --lambda_causal_contrast 0.5",
            "model_dir": "logs/BLiSS"
        },
        {
            "name": "unsupervised_intervention",
            "description": "无监督干预对齐 (创新): 干预非相似度作为对齐度量",
            "cmd": "python train.py --dataset BLiSS --max_epoch 30 --batch_size 64 --print_freq 50 --eval_freq 5 --disable_alignment_mask --lambda_causal_contrast 0.5 --enable_causal_alignment",
            "model_dir": "logs/BLiSS"
        }
    ]
    
    results = {}
    
    for i, exp in enumerate(experiments):
        print(f"\n\n{'#'*70}")
        print(f"# 实验 {i+1}/{len(experiments)}: {exp['name']}")
        print(f"# {exp['description']}")
        print('#'*70)
        
        # 备份现有模型
        backup_dir = os.path.join(base_dir, exp['name'])
        os.makedirs(backup_dir, exist_ok=True)
        
        # 运行训练
        returncode = run_command(exp['cmd'], exp['description'])
        
        if returncode != 0:
            print(f"⚠️ 实验 {exp['name']} 失败!")
            results[exp['name']] = {"status": "failed"}
            continue
        
        # 复制模型到备份目录
        if os.path.exists(f"{exp['model_dir']}/checkpoint"):
            os.system(f"xcopy /E /I /Y \"{exp['model_dir']}\\checkpoint\" \"{backup_dir}\\checkpoint\"")
        
        # 运行因果分析
        print(f"\n运行因果分析...")
        analysis_cmd = f"python bliss_causal_analysis.py --dataset BLiSS"
        run_command(analysis_cmd, f"分析 {exp['name']}")
        
        results[exp['name']] = {
            "status": "completed",
            "model_dir": backup_dir
        }
    
    # 保存实验配置
    config_path = os.path.join(base_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "experiments": experiments,
            "results": results
        }, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print("实验完成!")
    print(f"结果保存至: {base_dir}")
    print('='*70)
    
    # 汇总结果
    print("\n实验汇总:")
    for name, result in results.items():
        status = "✓ 成功" if result['status'] == 'completed' else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n请检查 {base_dir} 目录获取详细结果")
    print("接下来可以运行: python bliss_causal_analysis.py --compare_models 进行对比分析")


if __name__ == '__main__':
    main()
