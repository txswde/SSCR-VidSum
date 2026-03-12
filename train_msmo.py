import logging
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from models import *
from losses import *
from datasets import *
from utils import *


from rouge_score import rouge_scorer

# Counterfactual explainable summarization for CNN/Daily Mail
from counterfactual_summarization import CounterfactualSummarizationLoss

# Causal contrastive loss for BLiSS
from causal_contrastive_loss import CausalContrastiveLoss

# Causal alignment discovery for BLiSS (Innovation: Intervention Dissimilarity)
from causal_alignment_discovery import CausalAlignmentDiscovery

logger = logging.getLogger()

def train_msmo(args):
    batch_time = AverageMeter('time')
    data_time = AverageMeter('time')

    if args.dataset == 'BLiSS':
        model = Model_BLiSS(args=args)
    elif args.dataset in ['Daily_Mail', 'CNN']:
        model = Model_MSMO(args=args)

    model = model.to(args.device)
    calc_contrastive_loss = Dual_Contrastive_Loss().to(args.device)

    # Initialize counterfactual loss for CNN/Daily_Mail (no temporal alignment)
    cf_loss_fn = None
    if args.dataset in ['Daily_Mail', 'CNN'] and getattr(args, 'enable_cf_explanation', False):
        cf_loss_fn = CounterfactualSummarizationLoss(
            lambda_cf_contrast=args.lambda_cf_contrast,
            lambda_cf_effect=args.lambda_cf_effect,
            lambda_cf_sensitivity=getattr(args, 'lambda_cf_sensitivity', 0.2),
            enable_text_cf=True,
            enable_video_cf=True,
            cf_strategy=args.cf_strategy
        ).to(args.device)
        logger.info(f'Counterfactual explainability enabled for {args.dataset}')

    # Initialize causal loss for BLiSS dataset
    causal_loss_fn = None
    causal_alignment_discovery = None  # Innovation: Intervention Dissimilarity
    
    if args.dataset == 'BLiSS' and getattr(args, 'lambda_causal_contrast', 0) > 0:
        causal_loss_fn = {
            'contrast': CausalContrastiveLoss(
                temperature=0.1,
                margin=0.5,
                lambda_invariance=0.1
            ).to(args.device),
            'lambda_contrast': args.lambda_causal_contrast
        }
        logger.info(f'Causal contrastive loss enabled for BLiSS: '
                   f'lambda_contrast={args.lambda_causal_contrast}')
    
    # === Innovation: Causal Alignment Discovery (Intervention Dissimilarity) ===
    if args.dataset == 'BLiSS' and getattr(args, 'enable_causal_alignment', False):
        causal_alignment_discovery = CausalAlignmentDiscovery(
            lambda_diagonal=getattr(args, 'lambda_causal_diagonal', 1.0),
            lambda_sparsity=getattr(args, 'lambda_causal_sparsity', 0.1),
            temperature=1.0,
            normalize_alignment=True
        )
        logger.info(f'Causal Alignment Discovery enabled for BLiSS (Intervention Dissimilarity)')

    parameters = [p for p in model.parameters() if p.requires_grad] + \
                    [p for p in calc_contrastive_loss.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs('{}/checkpoint'.format(args.model_dir), exist_ok=True)

    args.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)

    max_train_R1 = max_train_R2 = max_train_RL = max_train_cos = 0
    max_val_R1 = max_val_R2 = max_val_RL = max_val_cos = 0
    best_val_epoch = 0

    if args.dataset in ['Daily_Mail', 'CNN']:
        dataset_name = 'MSMODataset'
    elif args.dataset in ['BLiSS']:
        dataset_name = 'BLiSSDataset'

    train_set = eval(dataset_name)(mode='train', args=args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
                                                drop_last=False, pin_memory=True, 
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    val_set = eval(dataset_name)(mode='test', args=args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                                                drop_last=False, pin_memory=True, 
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)

    checkpoint_path = None
    if args.checkpoint and args.test:
        checkpoint_path = '{}/model_best_text.pt'.format(args.checkpoint)
        print(f"load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        val_R1, val_R2, val_RL, _ = evaluate_msmo(model, val_loader, args, epoch=0)

        checkpoint_path = '{}/model_best_video.pt'.format(args.checkpoint)
        print(f"load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        _, _, _, val_cos = evaluate_msmo(model, val_loader, args, epoch=0)

        logger.info(f'R1: {val_R1:.4f} R2: {val_R2:.4f} RL: {val_RL:.4f} Cos: {val_cos:.4f}')
        return val_R1, val_R2, val_RL, val_cos, best_val_epoch, max_train_R1, max_train_R2, max_train_RL, max_train_cos

    logger.info('\n' + str(model))

    for epoch in range(args.start_epoch, args.max_epoch):
        model.train()
        stats = AverageMeter('loss', 'text_loss', 'video_loss', 'inter_contrastive_loss', 'intra_contrastive_loss', 'cf_loss', 'causal_loss', 'R1', 'R2', 'RL', 'cos')

        data_length = len(train_loader)
        end = time.time()
        for k, (video_list, video_summ_list, text_list, \
                mask_video_list, mask_video_summ_list, mask_text_list, \
                video_label_list, text_label_list, article_segment_list, highlight_list, \
                video_to_text_mask_list, text_to_video_mask_list) in enumerate(train_loader):
            data_time.update(time=time.time() - end)

            batch_size = len(video_list)

            video = pad_sequence(video_list, batch_first=True)
            video_summ = pad_sequence(video_summ_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)

            mask_video = pad_sequence(mask_video_list, batch_first=True)
            mask_video_summ = pad_sequence(mask_video_summ_list, batch_first=True)
            mask_text = pad_sequence(mask_text_list, batch_first=True)
            
            video_label = pad_sequence(video_label_list, batch_first=True)
            text_label = pad_sequence(text_label_list, batch_first=True)

            for i in range(len(video_to_text_mask_list)):
                # For emergent alignment experiment: disable temporal alignment mask
                if getattr(args, 'disable_alignment_mask', False):
                    # Use full attention instead of temporal-aligned attention
                    # This tests if model can learn alignment without supervision
                    video_to_text_mask_list[i] = torch.ones_like(video_to_text_mask_list[i])
                    text_to_video_mask_list[i] = torch.ones_like(text_to_video_mask_list[i])
                video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
                text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

            video, video_summ, text = video.to(args.device), video_summ.to(args.device), text.to(args.device)
            mask_video, mask_video_summ, mask_text = mask_video.to(args.device), mask_video_summ.to(args.device), mask_text.to(args.device)
    
            video_label = video_label.to(args.device) #[B, T]
            text_label = text_label.to(args.device) #[B, T]

            pred_video, pred_text, contrastive_pairs = model(video=video, text=text, \
                                                                mask_video=mask_video, mask_text=mask_text, \
                                                                video_label=video_label, text_label=text_label, \
                                                                video_to_text_mask_list=video_to_text_mask_list, \
                                                                text_to_video_mask_list=text_to_video_mask_list)

            num_frame_selected = torch.sum(video_label, dim=-1)
            num_sentence_selected = torch.sum(text_label, dim=-1)

            mask_video_bool = mask_video.to(torch.bool)
            mask_video_summ_bool = mask_video_summ.to(torch.bool)
            mask_text_bool = mask_text.to(torch.bool)

            # select frames and sentences with top-k highest importance score as predicted video and text summary
            keyframe_index_list = []
            keysentence_index_list = []
            for i in range(batch_size):
                keyframe_index_list.append(torch.topk(pred_video[i, mask_video_bool[i]], k=num_frame_selected[i])[1].tolist())
                keysentence_index_list.append(torch.topk(pred_text[i, mask_text_bool[i]], k=num_sentence_selected[i])[1].tolist())

            text_loss = calc_cls_loss(pred_text, text_label, mask=mask_text)
            if args.dataset in ['Daily_Mail', 'BLiSS']:
                video_loss = calc_cls_loss(pred_video, video_label, mask=mask_video)
            else:
                video_loss = torch.zeros(1).to(text_loss)

            inter_contrastive_loss, intra_contrastive_loss = calc_contrastive_loss(contrastive_pairs)
            
            inter_contrastive_loss = inter_contrastive_loss * args.lambda_contrastive_inter
            intra_contrastive_loss = intra_contrastive_loss * args.lambda_contrastive_intra
            loss = video_loss + text_loss + inter_contrastive_loss + intra_contrastive_loss

            # Counterfactual explainability loss for CNN/Daily_Mail
            cf_loss = torch.zeros(1, device=args.device)
            if cf_loss_fn is not None:
                cf_losses = cf_loss_fn(
                    model=model,
                    video=video, text=text,
                    mask_video=mask_video, mask_text=mask_text,
                    video_label=video_label, text_label=text_label,
                    video_to_text_mask_list=video_to_text_mask_list,
                    text_to_video_mask_list=text_to_video_mask_list
                )
                cf_loss = cf_losses['total_cf_loss']
                loss = loss + cf_loss

            # Causal contrastive loss for BLiSS
            causal_loss = torch.zeros(1, device=args.device)
            if causal_loss_fn is not None:
                # Generate counterfactual text by zeroing random sentences
                text_cf_min = text.clone()
                text_cf_sev = text.clone()
                
                for b in range(batch_size):
                    num_valid = int(mask_text[b].sum().item())
                    if num_valid > 1:
                        # Minimal intervention: zero 1 random sentence
                        min_idx = torch.randint(0, num_valid, (1,)).item()
                        text_cf_min[b, min_idx] = 0
                        # Severe intervention: zero 30-50% of sentences
                        num_to_zero = max(1, int(num_valid * 0.4))
                        sev_indices = torch.randperm(num_valid)[:num_to_zero]
                        text_cf_sev[b, sev_indices] = 0
                
                # Get cls_text features from counterfactual forward passes
                # Use no_grad for CF passes (they serve as references, gradients flow through orig only)
                with torch.no_grad():
                    _, _, cf_min_pairs = model(
                        video=video, text=text_cf_min,
                        mask_video=mask_video, mask_text=mask_text,
                        video_label=video_label, text_label=text_label,
                        video_to_text_mask_list=video_to_text_mask_list,
                        text_to_video_mask_list=text_to_video_mask_list
                    )
                    _, _, cf_sev_pairs = model(
                        video=video, text=text_cf_sev,
                        mask_video=mask_video, mask_text=mask_text,
                        video_label=video_label, text_label=text_label,
                        video_to_text_mask_list=video_to_text_mask_list,
                        text_to_video_mask_list=text_to_video_mask_list
                    )
                
                # Use cls_text as the text representation: [B, 1, D]
                feat_orig = contrastive_pairs['cls_text']         # [B, 1, D] - has gradients
                feat_cf_min = cf_min_pairs['cls_text'].detach()   # [B, 1, D] - detached
                feat_cf_sev = cf_sev_pairs['cls_text'].detach()   # [B, 1, D] - detached
                
                contrast_losses = causal_loss_fn['contrast'](
                    feat_orig, feat_cf_min, feat_cf_sev
                )
                causal_loss = contrast_losses['total'] * causal_loss_fn['lambda_contrast']
                loss = loss + causal_loss

            # === Innovation: Causal Alignment Discovery (Intervention Dissimilarity) ===
            causal_align_loss = torch.zeros(1, device=args.device)
            if causal_alignment_discovery is not None and (epoch + 1) >= getattr(args, 'alignment_warmup_epochs', 5):
                # Only compute every N iterations to reduce overhead
                if (k + 1) % getattr(args, 'alignment_compute_freq', 10) == 0:
                    align_result = causal_alignment_discovery(
                        model=model,
                        video=video, text=text,
                        mask_video=mask_video, mask_text=mask_text,
                        video_label=video_label, text_label=text_label,
                        video_to_text_mask_list=video_to_text_mask_list,
                        text_to_video_mask_list=text_to_video_mask_list,
                        compute_loss=True,
                        evaluate=(k + 1) % (args.print_freq * 5) == 0  # Evaluate less frequently
                    )
                    causal_align_loss = align_result['total_causal_alignment_loss'] * getattr(args, 'lambda_causal_alignment', 0.5)
                    loss = loss + causal_align_loss
                    
                    # Log evaluation metrics occasionally
                    if 'evaluation' in align_result and (k + 1) % (args.print_freq * 5) == 0:
                        eval_metrics = align_result['evaluation']
                        logger.info(f'[Causal Alignment] Correlation: {eval_metrics["correlation"]:.4f}, '
                                   f'P@1: {eval_metrics["precision_at_1"]:.4f}, IoU: {eval_metrics["iou"]:.4f}')

            if args.dataset in ['Daily_Mail', 'BLiSS']:
                video_cos = calc_video_cos(video, video_summ, keyframe_index_list, mask_video_summ=mask_video_summ_bool, dataset=args.dataset)
            else:
                video_cos = 0
            text_R1, text_R2, text_RL = calc_text_rouge(article_segment_list, highlight_list, keysentence_index_list, dataset=args.dataset, rouge=args.rouge)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), text_loss=text_loss.item(), video_loss=video_loss.item(), 
                            inter_contrastive_loss=inter_contrastive_loss.item(), intra_contrastive_loss=intra_contrastive_loss.item(),
                            cf_loss=cf_loss.item() if isinstance(cf_loss, torch.Tensor) else cf_loss,
                            causal_loss=causal_loss.item() if isinstance(causal_loss, torch.Tensor) else causal_loss,
                            R1=text_R1, R2=text_R2, RL=text_RL, cos=video_cos)

            batch_time.update(time=time.time() - end)
            end = time.time()

            if (k + 1) % args.print_freq == 0:
                cf_loss_str = f'CF: {stats.cf_loss:.4f} ' if cf_loss_fn is not None else ''
                causal_loss_str = f'Causal: {stats.causal_loss:.4f} ' if causal_loss_fn is not None else ''
                logger.info(f'[Train] Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} LR: {args.lr:.4f} '
                            f'Time: {batch_time.time:.3f} Data: {data_time.time:.3f} '
                            f'Loss: {stats.text_loss:.4f}/{stats.video_loss:.4f}/{stats.inter_contrastive_loss:.4f}/{stats.intra_contrastive_loss:.4f}/{stats.loss:.4f} '
                            f'{cf_loss_str}{causal_loss_str}'
                            f'R1: {stats.R1:.4f} R2: {stats.R2:.4f} RL: {stats.RL:.4f} Cos: {stats.cos:.4f}')

        max_train_R1 = max(stats.R1, max_train_R1)
        max_train_R2 = max(stats.R2, max_train_R2)
        max_train_RL = max(stats.RL, max_train_RL)
        max_train_cos = max(stats.cos, max_train_cos)

        logger.info(f'[Train] Epoch: {epoch+1}/{args.max_epoch} '
                    f'R1: {stats.R1:.4f}/{max_train_R1:.4f} '
                    f'R2: {stats.R2:.4f}/{max_train_R2:.4f} '
                    f'RL: {stats.RL:.4f}/{max_train_RL:.4f} '
                    f'Cos: {stats.cos:.4f}/{max_train_cos:.4f}\n'
        )

        args.writer.add_scalar(f'Train/max_train_R1', max_train_R1, epoch+1)
        args.writer.add_scalar(f'Train/max_train_R2', max_train_R2, epoch+1)
        args.writer.add_scalar(f'Train/max_train_RL', max_train_RL, epoch+1)
        args.writer.add_scalar(f'Train/max_train_cos', max_train_cos, epoch+1)
        args.writer.add_scalar(f'Train/train_R1', stats.R1, epoch+1)
        args.writer.add_scalar(f'Train/train_R2', stats.R2, epoch+1)
        args.writer.add_scalar(f'Train/train_RL', stats.RL, epoch+1)
        args.writer.add_scalar(f'Train/train_cos', stats.cos, epoch+1)

        save_checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'max_val_R1': max_val_R1,
            'max_val_R2': max_val_R2,
            'max_val_RL': max_val_RL,
            'max_val_cos': max_val_cos,
        }

        if (epoch + 1) % args.eval_freq == 0:
            val_R1, val_R2, val_RL, val_cos = evaluate_msmo(model, val_loader, args, epoch=epoch)
            max_val_R2 = max(val_R2, max_val_R2)
            max_val_RL = max(val_RL, max_val_RL)
            if max_val_R1 < val_R1:
                max_val_R1 = max(val_R1, max_val_R1)
                best_val_epoch = epoch + 1
                torch.save(save_checkpoint, '{}/checkpoint/model_best_text.pt'.format(args.model_dir))
            if max_val_cos < val_cos:
                max_val_cos = max(val_cos, max_val_cos)
                torch.save(save_checkpoint, '{}/checkpoint/model_best_video.pt'.format(args.model_dir))
            
            logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} '
                        f'R1: {val_R1:.4f}/{max_val_R1:.4f} '
                        f'R2: {val_R2:.4f}/{max_val_R2:.4f} '
                        f'RL: {val_RL:.4f}/{max_val_RL:.4f} '
                        f'Cos: {val_cos:.4f}/{max_val_cos:.4f}\n\n'
            )

            args.writer.add_scalar(f'Val/max_val_R1', max_val_R1, epoch+1)
            args.writer.add_scalar(f'Val/max_val_R2', max_val_R2, epoch+1)
            args.writer.add_scalar(f'Val/max_val_RL', max_val_RL, epoch+1)
            args.writer.add_scalar(f'Val/max_val_cos', max_val_cos, epoch+1)
            args.writer.add_scalar(f'Val/val_R1', val_R1, epoch+1)
            args.writer.add_scalar(f'Val/val_R2', val_R2, epoch+1)
            args.writer.add_scalar(f'Val/val_RL', val_RL, epoch+1)
            args.writer.add_scalar(f'Val/val_cos', val_cos, epoch+1)

        args.writer.add_scalar(f'Train/loss', stats.loss, epoch+1)
        args.writer.add_scalar(f'Train/text_loss', stats.text_loss, epoch+1)
        args.writer.add_scalar(f'Train/video_loss', stats.video_loss, epoch+1)

    return max_val_R1, max_val_R2, max_val_RL, max_val_cos, best_val_epoch, \
            max_train_R1, max_train_R2, max_train_RL, max_train_cos


@torch.no_grad()
def evaluate_msmo(model, val_loader, args, epoch=None, mode='train'):
    stats = AverageMeter('R1', 'R2', 'RL', 'cos')
    data_length = len(val_loader)

    model.eval()
    for k, (video_list, video_summ_list, text_list, \
            mask_video_list, mask_video_summ_list, mask_text_list, \
            video_label_list, text_label_list, article_segment_list, highlight_list, \
            video_to_text_mask_list, text_to_video_mask_list) in enumerate(val_loader):

        batch_size = len(video_list)
        
        video = pad_sequence(video_list, batch_first=True)
        video_summ = pad_sequence(video_summ_list, batch_first=True)
        text = pad_sequence(text_list, batch_first=True)

        mask_video = pad_sequence(mask_video_list, batch_first=True)
        mask_video_summ = pad_sequence(mask_video_summ_list, batch_first=True)
        mask_text = pad_sequence(mask_text_list, batch_first=True)
        
        video_label = pad_sequence(video_label_list, batch_first=True)
        text_label = pad_sequence(text_label_list, batch_first=True)

        video, video_summ, text = video.to(args.device), video_summ.to(args.device), text.to(args.device)
        mask_video, mask_video_summ, mask_text = mask_video.to(args.device), mask_video_summ.to(args.device), mask_text.to(args.device)
        
        video_label = video_label.to(args.device) #[B, T]
        text_label = text_label.to(args.device) #[B, T]

        for i in range(len(video_to_text_mask_list)):
            # For emergent alignment experiment: disable temporal alignment mask
            if getattr(args, 'disable_alignment_mask', False):
                video_to_text_mask_list[i] = torch.ones_like(video_to_text_mask_list[i])
                text_to_video_mask_list[i] = torch.ones_like(text_to_video_mask_list[i])
            video_to_text_mask_list[i] = video_to_text_mask_list[i].to(args.device)
            text_to_video_mask_list[i] = text_to_video_mask_list[i].to(args.device)

        pred_video, pred_text, contrastive_pairs = model(video=video, text=text, \
                                                            mask_video=mask_video, mask_text=mask_text, \
                                                            video_label=video_label, text_label=text_label, \
                                                            video_to_text_mask_list=video_to_text_mask_list, \
                                                            text_to_video_mask_list=text_to_video_mask_list)

        num_frame_selected = torch.sum(video_label, dim=-1)
        num_sentence_selected = torch.sum(text_label, dim=-1)

        mask_video_bool = mask_video.to(torch.bool)
        mask_video_summ_bool = mask_video_summ.to(torch.bool)
        mask_text_bool = mask_text.to(torch.bool)
        keyframe_index_list = []
        keysentence_index_list = []
        for i in range(batch_size):
            keyframe_index_list.append(torch.topk(pred_video[i, mask_video_bool[i]], k=num_frame_selected[i])[1].tolist())
            keysentence_index_list.append(torch.topk(pred_text[i, mask_text_bool[i]], k=num_sentence_selected[i])[1].tolist())

        if args.dataset in ['Daily_Mail', 'BLiSS']:
            video_cos = calc_video_cos(video, video_summ, keyframe_index_list, mask_video_summ=mask_video_summ_bool, dataset=args.dataset)
        else:
            video_cos = 0
        text_R1, text_R2, text_RL = calc_text_rouge(article_segment_list, highlight_list, keysentence_index_list, dataset=args.dataset, rouge=args.rouge)

        stats.update(R1=text_R1, R2=text_R2, RL=text_RL, cos=video_cos)
        
        if (k + 1) % args.print_freq == 0:
            logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} '
                        f'R1: {stats.R1:.4f} R2: {stats.R2:.4f} RL: {stats.RL:.4f} Cos: {stats.cos:.4f}')
    return stats.R1, stats.R2, stats.RL, stats.cos
