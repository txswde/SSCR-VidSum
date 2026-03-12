import json 
import argparse
import numpy as np

_DATASET_HYPER_PARAMS = {
    "SumMe":{
        "lr":5e-4,
        "weight_decay": 1e-3,
        "max_epoch": 200,
        "batch_size": 4,
        "seed": 666,

        "num_input_video": 1024,
        "num_input_text": 768,
        "num_hidden": 128,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.5,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.1,
        "lambda_contrastive_intra": 3.0,
        "ratio": 16,
        
        # Causal learning parameters
        "lambda_causal_contrast": 0.5,
        "lambda_causal_effect": 0.3,
        "lambda_alignment": 1.0,
        "lambda_text_necessity": 1.0,
        "cf_strategy": "mixed",
        
        # Innovation: Causal Alignment Discovery (Intervention Dissimilarity)
        "disable_alignment_mask": False,  # Set True for unsupervised alignment
        "enable_causal_alignment": False,  # Set True to enable intervention dissimilarity
        "lambda_causal_alignment": 0.5,
        "lambda_causal_diagonal": 1.0,
        "lambda_causal_sparsity": 0.1,
        "alignment_warmup_epochs": 3,
        "alignment_compute_freq": 5,
    },

    "TVSum":{
        "lr":1e-3,
        "weight_decay": 1e-5,
        "max_epoch": 300,
        "batch_size": 4,
        "seed": 666,

        "num_input_video": 1024,
        "num_input_text": 768,
        "num_hidden": 128,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.5,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.1,
        "lambda_contrastive_intra": 1.0,
        "ratio": 16,
        
        # Causal learning parameters
        "lambda_causal_contrast": 0.5,
        "lambda_causal_effect": 0.3,
        "lambda_alignment": 1.0,
        "lambda_text_necessity": 1.0,
        "cf_strategy": "mixed",
        
        # Innovation: Causal Alignment Discovery (Intervention Dissimilarity)
        "disable_alignment_mask": False,  # Set True for unsupervised alignment
        "enable_causal_alignment": False,  # Set True to enable intervention dissimilarity
        "lambda_causal_alignment": 0.5,
        "lambda_causal_diagonal": 1.0,
        "lambda_causal_sparsity": 0.1,
        "alignment_warmup_epochs": 3,
        "alignment_compute_freq": 5,
    },

    "BLiSS":{
        "lr":1e-3,
        "weight_decay": 1e-7,
        "max_epoch": 50,
        "batch_size": 64,
        "seed": 12345,

        "num_input_video": 512,
        "num_input_text": 768,
        "num_hidden": 128,
        "num_layers": 6,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.01,
        "lambda_contrastive_intra": 0.001,
        "ratio": 4,
        
        # Causal learning parameters
        "lambda_causal_contrast": 0.3,
        "lambda_causal_effect": 0.2,
        "cf_strategy": "mixed",
        
        # Emergent alignment experiment
        "disable_alignment_mask": False,  # Set True for unsupervised alignment
        
        # Innovation: Causal Alignment Discovery (Intervention Dissimilarity)
        "enable_causal_alignment": False,  # Set True to enable intervention dissimilarity
        "lambda_causal_alignment": 0.5,
        "lambda_causal_diagonal": 1.0,
        "lambda_causal_sparsity": 0.1,
        "alignment_warmup_epochs": 5,
        "alignment_compute_freq": 10,
        "data_ratio": 1.0,
    },

    "Daily_Mail":{
        "lr":2e-4,
        "weight_decay": 1e-7,
        "max_epoch": 100,
        "batch_size": 4,
        "seed": 12345,

        "num_input_video": 2048,
        "num_input_text": 768,
        "num_hidden": 256,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.001,
        "lambda_contrastive_intra": 0.001,
        "ratio": 8,
        
        # Counterfactual explainable summarization (NO alignment loss)
        "lambda_cf_contrast": 0.5,
        "lambda_cf_effect": 0.3,
        "lambda_cf_sensitivity": 0.2,
        "cf_strategy": "mixed",
        "enable_cf_explanation": True,
    },

    "CNN":{
        "lr":2e-4,
        "weight_decay": 1e-5,
        "max_epoch": 100,
        "batch_size": 4,
        "seed": 12345,

        "num_input_video": 2048,
        "num_input_text": 768,
        "num_hidden": 256,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.0,
        "lambda_contrastive_intra": 0.0,
        "ratio": 0,
        
        # Counterfactual explainable summarization (NO alignment loss)
        "lambda_cf_contrast": 0.5,
        "lambda_cf_effect": 0.3,
        "lambda_cf_sensitivity": 0.2,
        "cf_strategy": "mixed",
        "enable_cf_explanation": True,
    },

} 

def build_args():
    parser = argparse.ArgumentParser("This script is used for the multimodal summarization task.")

    parser.add_argument('--dataset', type=str, default=None, choices=['TVSum', 'SumMe', 'BLiSS', 'Daily_Mail', 'CNN'])
    parser.add_argument('--data_root', type=str, default='data')

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='logs')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--nms_thresh', type=float, default=0.4)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='ratio of training data to use (0.0-1.0), for ablation experiments')

    # inference
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--test', default=False, action='store_true', help='test mode')

    # common model config
    parser.add_argument('--num_input_video', type=int, default=1024)
    parser.add_argument('--num_input_text', type=int, default=768)
    parser.add_argument('--num_feature', type=int, default=512)
    parser.add_argument('--num_hidden', type=int, default=128)
    
    # transformer config
    parser.add_argument('--dropout_video', type=float, default=0.1, help='pre_drop for video')
    parser.add_argument('--dropout_text', type=float, default=0.1, help='pre_drop for text')
    parser.add_argument('--dropout_attn', type=float, default=0.1, help='dropout for attention operation in transformer')
    parser.add_argument('--dropout_fc', type=float, default=0.5, help='dropout for final classification')
    parser.add_argument('--num_layers', type=int, default=1)

    # contrastive loss
    parser.add_argument('--lambda_contrastive_inter', type=float, default=0.0)
    parser.add_argument('--lambda_contrastive_intra', type=float, default=0.0)
    parser.add_argument('--ratio', type=int, default=16)
    
    # causal learning parameters
    parser.add_argument('--lambda_causal_contrast', type=float, default=0.5, 
                        help='weight for causal contrastive loss')
    parser.add_argument('--lambda_causal_effect', type=float, default=0.3,
                        help='weight for causal effect loss')
    parser.add_argument('--lambda_alignment', type=float, default=1.0,
                        help='weight for temporal alignment loss')
    parser.add_argument('--lambda_text_necessity', type=float, default=1.0,
                        help='weight for text necessity loss')
    parser.add_argument('--cf_strategy', type=str, default='mixed',
                        choices=['zero', 'noise', 'shuffle', 'mask', 'mixed'],
                        help='counterfactual generation strategy')
    parser.add_argument('--disable_alignment_mask', action='store_true',
                        help='disable temporal alignment mask for emergent alignment experiment')
    
    # Innovation: Causal Alignment Discovery (Intervention Dissimilarity) - BLiSS only
    parser.add_argument('--enable_causal_alignment', action='store_true', default=False,
                        help='enable causal alignment discovery via intervention dissimilarity')
    parser.add_argument('--lambda_causal_alignment', type=float, default=0.5,
                        help='weight for causal alignment discovery loss')
    parser.add_argument('--lambda_causal_diagonal', type=float, default=1.0,
                        help='weight for diagonal regularization in causal alignment')
    parser.add_argument('--lambda_causal_sparsity', type=float, default=0.1,
                        help='weight for sparsity regularization in causal alignment')
    parser.add_argument('--alignment_warmup_epochs', type=int, default=5,
                        help='warmup epochs before enabling causal alignment loss')
    parser.add_argument('--alignment_compute_freq', type=int, default=10,
                        help='compute causal alignment every N iterations')
    
    
    # Counterfactual explainable summarization parameters (for CNN/Daily_Mail)
    parser.add_argument('--lambda_cf_contrast', type=float, default=0.5,
                        help='weight for counterfactual contrast loss')
    parser.add_argument('--lambda_cf_effect', type=float, default=0.3,
                        help='weight for counterfactual causal effect loss')
    parser.add_argument('--lambda_cf_sensitivity', type=float, default=0.2,
                        help='weight for counterfactual sensitivity loss')
    parser.add_argument('--enable_cf_explanation', action='store_true', default=False,
                        help='enable counterfactual explainability for CNN/Daily_Mail')
    
    args = parser.parse_args()
    
    # Get default values from dataset config
    dataset_config = _DATASET_HYPER_PARAMS[args.dataset]
    
    # Helper function: only override if using default value
    def set_if_default(arg_name, config_key=None):
        config_key = config_key or arg_name
        default_value = parser.get_default(arg_name)
        current_value = getattr(args, arg_name)
        # Only override if current value equals parser default (not set by user)
        if current_value == default_value and config_key in dataset_config:
            setattr(args, arg_name, dataset_config[config_key])
    
    # Apply dataset-specific defaults (only if not overridden by command line)
    set_if_default('lr')
    set_if_default('weight_decay')
    set_if_default('max_epoch')
    set_if_default('batch_size')
    set_if_default('seed')
    
    set_if_default('num_input_video')
    set_if_default('num_input_text')
    set_if_default('num_hidden')
    set_if_default('num_layers')
    
    set_if_default('dropout_video')
    set_if_default('dropout_text')
    set_if_default('dropout_attn')
    set_if_default('dropout_fc')
    
    set_if_default('lambda_contrastive_inter')
    set_if_default('lambda_contrastive_intra')
    set_if_default('ratio')
    
    # Causal learning parameters
    set_if_default('lambda_causal_contrast')
    set_if_default('lambda_causal_effect')
    set_if_default('lambda_alignment')
    
    # Innovation: Causal Alignment Discovery (for BLiSS, SumMe, TVSum)
    set_if_default('disable_alignment_mask')
    set_if_default('enable_causal_alignment')
    set_if_default('lambda_causal_alignment')
    set_if_default('lambda_causal_diagonal')
    set_if_default('lambda_causal_sparsity')
    set_if_default('alignment_warmup_epochs')
    set_if_default('alignment_compute_freq')
    set_if_default('data_ratio')
    
    # Counterfactual explainable summarization (CNN/Daily_Mail)
    set_if_default('lambda_cf_contrast')
    set_if_default('lambda_cf_effect')
    set_if_default('lambda_cf_sensitivity')
    set_if_default('enable_cf_explanation')
    set_if_default('cf_strategy')
    
    return args

def get_arguments() -> argparse.Namespace:
    args = build_args()

    args.model_dir = f'{args.model_dir}/{args.dataset}'
    if len(args.suffix) > 0:
        args.model_dir += f'_{args.suffix}'
    return args

