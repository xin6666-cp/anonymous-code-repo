# main_github.py
import os
import json
import logging
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import GPT2TokenizerFast


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0', ''):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


# =============================================================================
# args
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='CRM + SPR main launcher')

    # ---- 基础 ----
    parser.add_argument('--data_name', type=str, default='mimic_cxr')
    parser.add_argument('--version', type=str, default='best')
    parser.add_argument('--task', type=str, default='report-generation-gpt2')
    parser.add_argument('--phase', type=str, default='finetune',
                        choices=['pretrain', 'finetune', 'inference', 'test'])
    parser.add_argument('--ann_path', type=str, required=True)
    parser.add_argument('--view_position_dict', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--encoder_max_length', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--is_save_checkpoint', type=str, default='yes')

    # ---- 加载 / 续训 ----
    parser.add_argument('--load', type=str, default='',
                        help='Stage 1 checkpoint path; weights-only partial load')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume full Stage 2 training state')

    parser.add_argument('--ckpt_zoo_dir', type=str, default='')
    parser.add_argument('--temporal_fusion_num_blocks', type=int, default=3)
    parser.add_argument('--perceiver_num_blocks', type=int, default=3)
    parser.add_argument('--num_latents', type=int, default=128)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--pt_lr', type=float, default=5.0e-6)
    parser.add_argument('--ft_lr', type=float, default=5.0e-5)
    parser.add_argument('--monitor_metric', type=str, default='RCB')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=12)

    parser.add_argument('--monitor_mode', type=str, default='max', choices=['max', 'min'])
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--text_encoder_num_blocks', type=int, default=6)
    parser.add_argument('--print_step', type=int, default=1000)
    parser.add_argument('--num_beams', type=int, default=4)

    # ---- 预训练模型路径 ----
    parser.add_argument('--rad_dino_path', type=str, default='./checkpoints/pretrained/rad-dino')
    parser.add_argument('--cxr_bert_path', type=str, default='./checkpoints/pretrained/BiomedVLP-CXR-BERT-specialized')
    parser.add_argument('--distilgpt2_path', type=str, default='./checkpoints/pretrained/distilgpt2')
    parser.add_argument('--chexbert_path', type=str, default='./checkpoints/evaluator/chexbert.pth')
    parser.add_argument('--bert_path', type=str, default='./checkpoints/pretrained/bert-base-uncased')

    # ---- 模块开关 ----
    parser.add_argument('--use_crm', type=str2bool, default=True,
                        help='Comparison Reasoning Module')
    parser.add_argument('--use_spr', type=str2bool, default=True,
                        help='Soft Prior Registration (CSPA + ECE) before CRM')

    # ---- CRM 超参 ----
    parser.add_argument('--num_comparison_queries', type=int, default=24)
    parser.add_argument('--crm_num_perceiver_layers', type=int, default=2)
    parser.add_argument('--lambda_tc', type=float, default=0.1,
                        help='Weight of CRM auxiliary change-classification loss')

    # ---- SPR 超参 ----
    parser.add_argument('--spr_dropout', type=float, default=0.1)
    parser.add_argument('--spr_init_alpha', type=float, default=0.1,
                        help='CSPA 初始对齐强度 alpha; 内部以 logit 存储, sigmoid 还原')

    # ---- 其他 ----
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32')

    return parser.parse_args()


# =============================================================================
# logger
# =============================================================================
def setup_logger_fn(save_dir, exp_name):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'{exp_name}.log')
    mylog = logging.getLogger(exp_name)
    mylog.setLevel(logging.INFO)
    mylog.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); mylog.addHandler(fh)
    ch = logging.StreamHandler();       ch.setFormatter(fmt); mylog.addHandler(ch)
    mylog.propagate = False
    return mylog, log_path


# =============================================================================
# Tokenizer (50264 vocab)
# =============================================================================
def build_tokenizer(args):
    """3 special (pad/sep/cls) + 4 domain token, distilgpt2 50257 -> 50264.
    顺序/键/值不可改, 否则与 Stage 1 ckpt 的 vocab 行序错位。"""
    tokenizer = GPT2TokenizerFast.from_pretrained(args['distilgpt2_path'])
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
    })
    tokenizer.add_tokens(['[INDICATION]', '[HISTORY]',
                          '[Similar Cases]', '[FINDINGS]'])
    return tokenizer


# =============================================================================
# Load Stage 1 weights
# =============================================================================
# 预期内的 missing key 前缀 (Stage 2 新增模块, Stage 1 里没有):
EXPECTED_MISSING_KW = (
    'text_decoder',
    'crm',
    'soft_prior_registration',
    'ln_pre_prefix',
)


def load_stage1_weights(model, ckpt_path, mylog):
    """严格 shape 匹配加载 Stage 1 权重; 不匹配的 skip, 不做 truncate/extend。
    新模块 (CRM/SPR/decoder/ln_pre_prefix) 保留默认初始化。"""
    if not ckpt_path:
        mylog.info('[load] No --load specified, training from scratch.')
        return
    if not os.path.exists(ckpt_path):
        mylog.warning(f'[load] checkpoint not found: {ckpt_path}, training from scratch.')
        return

    mylog.info(f'[load] loading Stage 1 weights from: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    pre_state = ckpt.get('state_dict', ckpt)
    cur_state = model.state_dict()

    # vocab 维度自检
    for k in ['text_encoder.embeddings.word_embeddings.weight']:
        s_pre = tuple(pre_state[k].shape) if k in pre_state else None
        s_cur = tuple(cur_state[k].shape) if k in cur_state else None
        if s_pre is not None and s_cur is not None and s_pre != s_cur:
            mylog.warning(f'[load] vocab shape mismatch: ckpt={s_pre} vs model={s_cur}, key SKIPPED.')

    valid_state = {
        k: v for k, v in pre_state.items()
        if k in cur_state and v.shape == cur_state[k].shape
    }
    skipped_keys = [k for k in pre_state.keys() if k not in valid_state]
    missing_in_ckpt = [k for k in cur_state.keys() if k not in pre_state]

    cur_state.update(valid_state)
    model.load_state_dict(cur_state)

    mylog.info(
        f'[load] loaded={len(valid_state)}  skipped_in_ckpt={len(skipped_keys)}  '
        f'missing_in_ckpt={len(missing_in_ckpt)}')

    unexpected_missing = [
        k for k in missing_in_ckpt
        if not any(kw in k for kw in EXPECTED_MISSING_KW)
    ]
    if unexpected_missing:
        mylog.warning(f'[load] {len(unexpected_missing)} unexpected missing keys (not new-module whitelist):')
        for k in unexpected_missing[:20]:
            mylog.warning(f'        - {k}')
    else:
        mylog.info('[load] all missing keys are new modules (CRM/SPR/decoder/ln_pre_prefix). OK.')


# =============================================================================
# main
# =============================================================================
def main():
    args_ns = parse_args()
    args = vars(args_ns)
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime('%Y_%m_%d_%H')
    exp_tag = f"{args['version']}_{timestamp}"
    project_dir = os.path.join(args['save_dir'], args['data_name'], args['phase'], exp_tag)
    os.makedirs(project_dir, exist_ok=True)
    args['project_name'] = project_dir

    pl.seed_everything(args['seed'], workers=True)

    mylog, _ = setup_logger_fn(project_dir, 'train')
    mylog.info(f"Phase={args['phase']}  Data={args['data_name']}  Project={project_dir}")
    mylog.info(f"Modules: use_crm={args['use_crm']}  use_spr={args['use_spr']}")
    mylog.info(f"--load = {args['load'] or '(none)'}  --resume = {args['resume'] or '(none)'}")

    # ===== Tokenizer =====
    tokenizer = build_tokenizer(args)
    mylog.info(f'tokenizer vocab_size = {len(tokenizer)} (expected 50264)')
    if len(tokenizer) != 50264:
        mylog.warning(f'[tokenizer] vocab_size is {len(tokenizer)} but expected 50264; vocab will mismatch Stage 1 ckpt.')

    # ===== Model =====
    from models.model_github import Alignment, TrainLanguageModel

    if args['phase'] == 'pretrain':
        model = Alignment(args=args, tokenizer=tokenizer, logger=mylog)
    else:
        model = TrainLanguageModel(args=args, tokenizer=tokenizer, logger=mylog)
        if args['resume']:
            mylog.info('--resume specified, skipping --load (full state will be restored)')
        else:
            load_stage1_weights(model, args['load'], mylog)

    # ===== Callbacks =====
    callbacks = [LearningRateMonitor(logging_interval='epoch')]
    if str(args['is_save_checkpoint']).lower() == 'yes':
        ckpt_dir = os.path.join(project_dir, 'checkpoint')
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir, filename='best_model',
            monitor=args['monitor_metric'], mode=args['monitor_mode'],
            save_top_k=1, save_last=True, verbose=False))

    tb_logger = TensorBoardLogger(save_dir=project_dir, name='tb')

    trainer = pl.Trainer(
        max_epochs=args['epochs'],
        devices=args['num_gpus'] if torch.cuda.is_available() else 1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=args['print_step'],
        enable_progress_bar=False,
        precision=args['precision'],
        deterministic=False,
    )

    if args['phase'] in ('pretrain', 'finetune'):
        resume_path = args['resume'].strip() if args['resume'] else ''
        if resume_path:
            trainer.fit(model, ckpt_path=resume_path)
        else:
            trainer.fit(model)
        if str(args['is_save_checkpoint']).lower() == 'yes':
            trainer.test(model, ckpt_path='best')
    else:
        ckpt_path = args['load'] if args['load'] and os.path.exists(args['load']) else None
        trainer.test(model, ckpt_path=ckpt_path) if ckpt_path else trainer.test(model)


if __name__ == '__main__':
    main()

