# model_github.py
import copy
import json
import math
import re
from typing import Dict

import torch
import numpy as np
import torchmetrics
import transformers
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities import rank_zero_only
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2TokenizerFast, AutoModel, AutoConfig, AutoImageProcessor

from models.bert_model import TemporalFusion
from models.perceiver_pytorch import Perceiver
from tools.metrics.chexbert import F1CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from tools.dataset_github import (AlignDataset, FinetuneDataset,
                                  AlignCollateFn, FinetuneCollateFn)
# =============================================================================
#  Stage 1 :  Alignment  (unchanged)
# =============================================================================
class Alignment(pl.LightningModule):
    def __init__(self, args: Dict, tokenizer: GPT2TokenizerFast, logger, **kwargs):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.val_min_losses = {"epoch": -1, 'loss': 1000}
        self.val_best_scores = {"best_epoch": -1, "best_monitor_metric": -1.0}

        self.train_loss_metric = {'loss': torchmetrics.MeanMetric().to(args['device'])}
        self.val_loss_metric   = {'loss': torchmetrics.MeanMetric().to(args['device'])}
        self.test_loss_metric  = {'loss': torchmetrics.MeanMetric().to(args['device'])}

        # Image Encoder (frozen)
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder   = AutoModel.from_pretrained(args['rad_dino_path'])
        self.image_encoder.config.output_hidden_states = True
        image_dim = self.image_encoder.config.hidden_size
        self.freeze_parameters(self.image_encoder)

        # Text Encoder
        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection heads
        self.image_projection = VisualProjectionHead(image_dim, args['hidden_size'] // 2, args['hidden_size'])
        self.text_projection  = ProjectionHead(text_dim, args['hidden_size'] // 2, args['hidden_size'])

        # learnable position embeddings
        self.vp2id = json.load(open(args['view_position_dict']))
        self.vp_pos_embed   = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=True)
        self.temp_pos_embed = nn.Parameter(torch.randn(2, 1, args['hidden_size']), requires_grad=True)
        self.logit_scale    = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        self.layer_norm     = nn.LayerNorm(args['hidden_size'])

        # temporal fusion
        self.temporal_fusion = TemporalFusion(
            args['hidden_size'], args['temporal_fusion_num_blocks'],
            heads=args['num_heads'], dim_head=args['hidden_size'] // 4,
            mlp_dim=args['hidden_size'])

        # Perceiver
        self.perceiver = Perceiver(
            byte_dim=args['hidden_size'],
            depth=args['perceiver_num_blocks'],
            num_latents=args['num_latents'],
            latent_dim=args['hidden_size'],
            cross_heads=8, latent_heads=8,
            cross_dim_head=64, latent_dim_head=64,
            attn_dropout=0., ff_dropout=0.,
            weight_tie_layers=False, self_per_cross_attn=1)

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'], config=enc_config,
            ignore_mismatched_sizes=True, trust_remote_code=True)

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = AlignDataset(self.args, 'train', self.tokenizer)
            self.val_set   = AlignDataset(self.args, 'val',   self.tokenizer)
            print("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:
            self.test_set = AlignDataset(self.args, 'test', self.tokenizer)
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        collate_fn = AlignCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(self.train_set, batch_size=self.args['batch_size'],
                          num_workers=self.args['num_workers'], shuffle=True,
                          collate_fn=collate_fn, drop_last=True)

    def val_dataloader(self):
        collate_fn = AlignCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(self.val_set, batch_size=self.args['batch_size'],
                          num_workers=self.args['num_workers'], shuffle=False,
                          collate_fn=collate_fn, drop_last=False)

    def test_dataloader(self):
        collate_fn = AlignCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(self.test_set, batch_size=self.args['batch_size'],
                          num_workers=self.args['num_workers'], shuffle=False,
                          collate_fn=collate_fn, drop_last=False)

    def configure_optimizers(self):
        all_parameters = [p for p in self.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(all_parameters, lr=self.args['pt_lr'])
        lr_scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                         factor=0.1, patience=self.args['patience'])
        return {"optimizer": optimiser,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'monitor': self.args['monitor_metric'], 'frequency': 1}}

    def tokenization(self, text, device):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=False,
                                max_length=self.args['max_length'], truncation=True)
        inputs['input_ids']      = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        return inputs

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        labels = labels / labels.sum(1, keepdim=True)
        del patient_ids
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        global_text_embed  = F.normalize(global_text_embed,  dim=-1, p=2)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * global_image_embed @ global_text_embed.t()
        logits_per_text  = logits_per_image.t()
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text  = F.cross_entropy(logits_per_text,  labels)
        return (loss_image + loss_text) / 2.0

    def image_encoder_forward(self, images):
        with torch.no_grad():
            outputs = self.image_encoder(images)
            last_hidden_state = outputs['last_hidden_state']
            hidden_states = torch.stack(outputs['hidden_states'][1:], dim=1)
        return hidden_states, last_hidden_state

    def obtain_spatio_temporal_visual_features(self, current_study, prior_study=None):
        _, last_hidden_state = self.image_encoder_forward(current_study['image'])
        image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in current_study['view_position']]
        cur_vis_feat = torch.cat(image_pos_embed, dim=0) + last_hidden_state
        cur_vis_feat = self.image_projection(cur_vis_feat)
        cur_temporal_embed = self.temp_pos_embed[0].repeat(cur_vis_feat.shape[0], 1, 1)
        cur_vis_feat = cur_vis_feat + cur_temporal_embed
        spatio_temp_feat = torch.empty_like(cur_vis_feat).to(cur_vis_feat)
        if prior_study is not None:
            _, pri_last_hidden_state = self.image_encoder_forward(prior_study['image'])
            pri_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in prior_study['view_position']]
            pri_last_hidden_state = torch.cat(pri_pos_embed, dim=0) + pri_last_hidden_state
            pri_last_hidden_state = self.image_projection(pri_last_hidden_state)
            pri_temporal_embed = self.temp_pos_embed[1].repeat(pri_last_hidden_state.shape[0], 1, 1)
            pri_last_hidden_state = pri_temporal_embed + pri_last_hidden_state
            has_pri_idx = prior_study['pri_idx']
            temp_visual_features = self.temporal_fusion(cur_vis_feat[has_pri_idx], pri_last_hidden_state)
            spatio_temp_feat[has_pri_idx] = temp_visual_features
            no_pri_idx = prior_study['no_pri_idx']
        else:
            no_pri_idx = list(range(cur_vis_feat.shape[0]))
        spatio_temp_feat[no_pri_idx] = self.layer_norm(cur_vis_feat[no_pri_idx])
        return spatio_temp_feat

    def obtain_textual_features(self, reports, device, return_attention_mask=False):
        inputs = self.tokenization(reports, device=device)
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])
        if not return_attention_mask:
            return text_embed
        return text_embed, inputs['attention_mask']

    def forward(self, current_study, reports, reference_reports, patient_ids, context, prior_study=None, mode='train'):
        spatio_temp_feat = self.obtain_spatio_temporal_visual_features(current_study, prior_study)
        context_embed = self.obtain_textual_features(context, spatio_temp_feat.device)
        context_latents = self.perceiver(context_embed)
        spatio_temp_latents = self.perceiver(spatio_temp_feat, latent=context_latents)
        encoder_outputs = torch.cat([context_latents, spatio_temp_latents], dim=1)
        text_embed = self.obtain_textual_features(reports, spatio_temp_feat.device)
        image_cls_embed = torch.mean(encoder_outputs, dim=1)
        instance_loss = self.global_alignment_loss(image_cls_embed, text_embed[:, 0, :], patient_ids)
        return {'loss': instance_loss}

    def training_step(self, batch, batch_idx):
        image_ids, patient_ids, reports = batch['image_ids'], batch['patient_ids'], batch['report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        reference_reports = batch['reference_report']
        loss_dict = self(current_study, reports, reference_reports, patient_ids, context, prior_study, mode='train')
        self.log_dict({f'train_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=len(reports), prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")
        for key, loss in loss_dict.items():
            self.train_loss_metric[f"{key}"].update(loss.detach())
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        image_ids, patient_ids, reports = batch['image_ids'], batch['patient_ids'], batch['report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        reference_reports = batch['reference_report']
        loss_dict = self(current_study, reports, reference_reports, patient_ids, context, prior_study, mode='val')
        self.log_dict({f'val_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports), prog_bar=False, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")
        for key, loss in loss_dict.items():
            self.val_loss_metric[f"{key}"].update(loss)

    def test_step(self, batch, batch_idx):
        image_ids, patient_ids, reports = batch['image_ids'], batch['patient_ids'], batch['report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        reference_reports = batch['reference_report']
        loss_dict = self(current_study, reports, reference_reports, patient_ids, context, prior_study, mode='test')
        self.log_dict({f'test_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports), prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, "
                          f"{cur_loss_item}")
        for key, loss in loss_dict.items():
            if f"{key}" in self.test_loss_metric:
                self.test_loss_metric[f"{key}"].update(loss)

    def on_train_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.train_loss_metric.items():
            avg_metric = metric.compute(); metric.reset(); cur_all_loss[key] = avg_metric
        self.log_dict({f'train_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True,
                      on_step=False, prog_bar=False)
        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################")

    def on_validation_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.val_loss_metric.items():
            avg_metric = metric.compute(); metric.reset(); cur_all_loss[key] = avg_metric
        self.log_dict({f'val_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False, prog_bar=False)
        if cur_all_loss['loss'] < self.val_min_losses["loss"]:
            self.val_min_losses = {**cur_all_loss, "epoch": self.current_epoch}
        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        best_loss_item = ', '.join([f"{k} = {v}" for k, v in self.val_min_losses.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current val loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
            f"best validation loss: {best_loss_item}\n")

    def on_test_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.test_loss_metric.items():
            avg_metric = metric.compute(); metric.reset(); cur_all_loss[key] = avg_metric
        self.log_dict({f'test_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False, prog_bar=False)
        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, test is over, current loss:"
            f"{cur_loss_item}\n")


# =============================================================================
#  Stage 2 :  TrainLanguageModel  ==  SPR  +  CRM
# =============================================================================
class TrainLanguageModel(pl.LightningModule):
    def __init__(self, args: Dict, tokenizer: GPT2TokenizerFast, logger, **kwargs):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger

        # ======= Module switches =======
        # use_crm: comparison reasoning over (cur, pri, delta)
        # use_spr: soft prior registration BEFORE crm; only meaningful if use_crm
        self.use_crm = bool(args.get('use_crm', True))
        self.use_spr = bool(args.get('use_spr', True)) and self.use_crm

        self.train_set = None
        self.val_set   = None
        self.test_set  = None
        self.val_best_scores = {"best_epoch": -1, "best_monitor_metric": -1.0}

        # ======= Loss metrics =======
        self.train_loss_metric      = torchmetrics.MeanMetric()
        self.train_loss_main_metric = torchmetrics.MeanMetric()
        self.train_loss_tc_metric   = torchmetrics.MeanMetric()

        self.val_coco_metrics  = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics  = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'], model_path=args['bert_path'],
            mbatch_size=16, exp_dir=args['project_name'])
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'], model_path=args['bert_path'],
            mbatch_size=16, exp_dir=args['project_name'])

        self.val_report_logger  = ReportLogger(exp_dir=args['project_name'], split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args['project_name'], split='test_reports')

        # ======= Image Encoder (frozen) =======
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder   = AutoModel.from_pretrained(args['rad_dino_path'])
        self.image_encoder.config.output_hidden_states = True
        image_dim = self.image_encoder.config.hidden_size
        self.freeze_parameters(self.image_encoder)

        # ======= Text Encoder (CXR-BERT) =======
        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # ======= Projections =======
        self.image_projection = VisualProjectionHead(image_dim, args['hidden_size'] // 2, args['hidden_size'])
        self.text_projection  = ProjectionHead(text_dim,  args['hidden_size'] // 2, args['hidden_size'])

        self.vp2id = json.load(open(args['view_position_dict']))
        self.vp_pos_embed   = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=False)
        self.temp_pos_embed = nn.Parameter(torch.randn(2, 1, args['hidden_size']),     requires_grad=False)
        self.logit_scale    = nn.Parameter(torch.ones([]) * np.log(1 / 0.07),          requires_grad=False)
        self.layer_norm     = nn.LayerNorm(args['hidden_size'])

        self.temporal_fusion = TemporalFusion(
            args['hidden_size'], args['temporal_fusion_num_blocks'],
            heads=args['num_heads'], dim_head=args['hidden_size'] // 4,
            mlp_dim=args['hidden_size'])

        self.perceiver = Perceiver(
            byte_dim=args['hidden_size'],
            depth=args['perceiver_num_blocks'],
            num_latents=args['num_latents'],
            latent_dim=args['hidden_size'],
            cross_heads=8, latent_heads=8,
            cross_dim_head=64, latent_dim_head=64,
            attn_dropout=0., ff_dropout=0.,
            weight_tie_layers=False, self_per_cross_attn=1)

        # 三路 LN: 对 context_latents / spatio_temp_latents / C 统一归一化后再 concat
        self.ln_pre_prefix = nn.LayerNorm(args['hidden_size'])

        # ======= Text Decoder =======
        self.text_decoder = self.build_text_decoder()

        # ======= SPR (在 CRM 之前) =======
        self.soft_prior_registration = None
        if self.use_spr:
            from models.soft_prior_registration import SoftPriorRegistration
            self.soft_prior_registration = SoftPriorRegistration(
                dim=args['hidden_size'],
                num_heads=args.get('num_heads', 8),
                dropout=args.get('spr_dropout', 0.1),
                init_alpha=args.get('spr_init_alpha', 0.1),
            )

        # ======= CRM =======
        # CRM 输出 num_change_classes=4 个 logits，对应类别索引 0/1/2/3。
        # 与之对应的原始标签映射关系（在 forward 的 loss 计算中执行）：
        #   原始 label 1 (new)      → CRM 类别索引 0
        #   原始 label 2 (worsened) → CRM 类别索引 1
        #   原始 label 3 (improved) → CRM 类别索引 2
        #   原始 label 4 (stable)   → CRM 类别索引 3
        #   原始 label 0/5          → ignore_index（不参与监督）
        self.crm = None
        if self.use_crm:
            from models.comparison_reasoning import ComparisonReasoningModule
            self.crm = ComparisonReasoningModule(
                dim=args['hidden_size'],
                num_comparison_queries=args.get('num_comparison_queries', 16),
                num_perceiver_layers=args.get('crm_num_perceiver_layers', 2),
                num_heads=args.get('num_heads', 8),
                num_change_classes=4)

        # ======= Loss weights =======
        self.lambda_tc = float(args.get('lambda_tc', 0.3))

        # CRM class weights (optional, for imbalanced datasets)
        # 顺序对应 CRM 类别索引 0/1/2/3，即 new/worsened/improved/stable
        ccw = args.get('change_class_weights', None)
        if ccw is not None and self.use_crm:
            dist = torch.tensor(ccw, dtype=torch.float32)
            inv = 1.0 / dist.clamp_min(1e-6)
            inv = inv * (len(inv) / inv.sum())
            self.register_buffer('change_class_weights', inv)
        else:
            self.change_class_weights = None


    # ---------------------------------------------------------------------
    # builders
    # ---------------------------------------------------------------------
    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'], config=enc_config,
            ignore_mismatched_sizes=True, trust_remote_code=True)

    def build_text_decoder(self):
        config = transformers.GPT2Config.from_pretrained(self.args['distilgpt2_path'])
        config.add_cross_attention = True
        config.is_decoder = True
        config.vocab_size = len(self.tokenizer)
        decoder = transformers.GPT2LMHeadModel(config=config)
        decoder.resize_token_embeddings(len(self.tokenizer))

        class DummyEncoder:
            main_input_name = 'dummy'
            class DummyConfig(PretrainedConfig):
                model_type = 'bert'
            config = DummyConfig()
            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size
            def forward(self, *args, **kwargs): pass
            def get_output_embeddings(cls): return None

        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)
        return Decoder()

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    # ---------------------------------------------------------------------
    # dataset / dataloader / optimizer
    # ---------------------------------------------------------------------
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = FinetuneDataset(self.args, 'train', self.tokenizer)
            self.val_set   = FinetuneDataset(self.args, 'test',  self.tokenizer)
            msg = "No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__())
            self.log_once(msg)
        if stage == "test" or stage is None:
            self.test_set = FinetuneDataset(self.args, 'test', self.tokenizer)
            msg = "No. of test examples: {}.".format(self.test_set.__len__())
            self.log_once(msg)

    def train_dataloader(self):
        collate_fn = FinetuneCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(self.train_set, batch_size=self.args['batch_size'],
                          num_workers=self.args['num_workers'], shuffle=True,
                          collate_fn=collate_fn, drop_last=True)

    def val_dataloader(self):
        collate_fn = FinetuneCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(self.val_set, batch_size=self.args['batch_size'],
                          num_workers=self.args['num_workers'], shuffle=False,
                          collate_fn=collate_fn, drop_last=False)

    def test_dataloader(self):
        collate_fn = FinetuneCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(self.test_set, batch_size=self.args['batch_size'],
                          num_workers=self.args['num_workers'], shuffle=False,
                          collate_fn=collate_fn, drop_last=False)

    def configure_optimizers(self):
        new_module_kw = ('text_decoder', 'crm', 'soft_prior_registration', 'ln_pre_prefix')
        ft_parameters, pt_parameters = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in new_module_kw):
                ft_parameters.append(param)
            else:
                pt_parameters.append(param)
        optimiser = torch.optim.AdamW(
            [{'params': pt_parameters, 'lr': self.args['pt_lr']},
             {'params': ft_parameters, 'lr': self.args['ft_lr']}])
        lr_scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                         factor=0.1, patience=self.args['patience'])
        return {"optimizer": optimiser,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'monitor': self.args['monitor_metric'], 'frequency': 1}}

    # ---------------------------------------------------------------------
    # tokenization helpers
    # ---------------------------------------------------------------------
    def tokenization(self, text, device, max_length):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=False,
                                max_length=max_length, truncation=True)
        inputs['input_ids']      = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        return inputs

    def obtain_reference_reports(self, text):
        inputs = self.tokenizer(text, padding=True, max_length=self.args['max_length'],
                                truncation=True, return_tensors='pt')
        ref_reports = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
        return ref_reports

    def obtain_decoder_input_ids(self, inputs):
        decoder_input_ids = inputs['input_ids']
        decoder_attention_mask = inputs['attention_mask'][:, :-1]
        label_ids = decoder_input_ids[:, 1:].detach().clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
        return decoder_input_ids, decoder_attention_mask, label_ids

    # ---------------------------------------------------------------------
    # Visual feature extraction
    # ---------------------------------------------------------------------
    def image_encoder_forward(self, images):
        with torch.no_grad():
            outputs = self.image_encoder(images)
            last_hidden_state = outputs['last_hidden_state']
            hidden_states = torch.stack(outputs['hidden_states'][1:], dim=1)
        return hidden_states, last_hidden_state

    def obtain_joint_visual_features_forward(self, current_study, prior_study=None):
        _, last_hidden_state = self.image_encoder_forward(current_study['image'])

        image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in current_study['view_position']]
        cur_vis_feat = torch.cat(image_pos_embed, dim=0) + last_hidden_state
        cur_vis_feat = self.image_projection(cur_vis_feat)   # pure (无 temp_pos)

        V_cur_pure = cur_vis_feat
        V_pri_pure = torch.zeros_like(V_cur_pure)
        prior_mask = torch.zeros(V_cur_pure.size(0),
                                 device=V_cur_pure.device, dtype=V_cur_pure.dtype)

        cur_temporal_embed = self.temp_pos_embed[0].repeat(cur_vis_feat.shape[0], 1, 1)
        cur_vis_feat_wt = cur_vis_feat + cur_temporal_embed
        spatio_temp_feat = torch.empty_like(cur_vis_feat_wt).to(cur_vis_feat_wt)

        if prior_study is not None:
            _, pri_last_hidden_state = self.image_encoder_forward(prior_study['image'])
            pri_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in prior_study['view_position']]
            pri_last_hidden_state = torch.cat(pri_pos_embed, dim=0) + pri_last_hidden_state
            pri_pure = self.image_projection(pri_last_hidden_state)

            has_pri_idx = prior_study['pri_idx']
            V_pri_pure[has_pri_idx] = pri_pure
            prior_mask[has_pri_idx] = 1.0

            pri_temporal_embed = self.temp_pos_embed[1].repeat(pri_pure.shape[0], 1, 1)
            pri_with_temp = pri_pure + pri_temporal_embed
            temp_visual_features = self.temporal_fusion(cur_vis_feat_wt[has_pri_idx], pri_with_temp)
            spatio_temp_feat[has_pri_idx] = temp_visual_features
            no_pri_idx = prior_study['no_pri_idx']
        else:
            no_pri_idx = list(range(cur_vis_feat_wt.shape[0]))
        spatio_temp_feat[no_pri_idx] = self.layer_norm(cur_vis_feat_wt[no_pri_idx])

        return spatio_temp_feat, V_cur_pure, V_pri_pure, prior_mask

    def obtain_textual_features(self, reports, device, return_attention_mask=False):
        inputs = self.tokenization(reports, device=device, max_length=self.args['encoder_max_length'])
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])
        if not return_attention_mask:
            return text_embed
        return text_embed, inputs['attention_mask']

    # ---------------------------------------------------------------------
    # generate (val/test)
    # ---------------------------------------------------------------------
    def generate(self, encoder_outputs):
        outputs = self.text_decoder.encoder_decoder.generate(
            max_length=self.args['max_length'],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.args['num_beams'],
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs)
        return outputs['sequences']

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, current_study, context, reference_reports=None, prior_study=None, mode='train',
                change_labels=None):
        # 1. 视觉特征 -- pure 路径给 SPR，spatio_temp_feat 路径给 Perceiver
        spatio_temp_feat, V_cur_pure, V_pri_pure, prior_mask = \
            self.obtain_joint_visual_features_forward(current_study, prior_study)

        # 2. 临床上下文
        context_embed = self.obtain_textual_features(context, spatio_temp_feat.device)

        # 3. Perceiver
        context_latents     = self.perceiver(context_embed)
        spatio_temp_latents = self.perceiver(spatio_temp_feat, latent=context_latents)

        # 4. SPR + CRM
        C = None
        change_logits = None
        if self.use_crm:
            V_cur_for_crm = V_cur_pure
            V_pri_for_crm = V_pri_pure
            V_delta_for_crm = None

            if self.soft_prior_registration is not None:
                V_cur_for_crm, V_pri_for_crm, V_delta_for_crm, _ = \
                    self.soft_prior_registration(
                        V_cur=V_cur_pure,
                        V_pri=V_pri_pure,
                        prior_mask=prior_mask,
                        return_attn=False,
                    )

            C, change_logits = self.crm(
                V_cur=V_cur_for_crm,
                V_pri=V_pri_for_crm,
                V_delta=V_delta_for_crm,
                prior_mask=prior_mask,
            )

        # 5. 三路 LN + concat -> decoder cross-attention inputs
        if self.use_crm:

            gate = prior_mask.view(C.size(0), 1, 1).to(dtype=C.dtype)
            C_normed = self.ln_pre_prefix(C) * gate
            encoder_outputs_tensor = torch.cat([
                self.ln_pre_prefix(context_latents),
                self.ln_pre_prefix(spatio_temp_latents),
                C_normed,
            ], dim=1)
        else:
            encoder_outputs_tensor = torch.cat([
                self.ln_pre_prefix(context_latents),
                self.ln_pre_prefix(spatio_temp_latents),
            ], dim=1)

        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=encoder_outputs_tensor)

        # =====================================================================
        # Train branch
        # =====================================================================
        if mode == 'train':
            assert reference_reports is not None, "train 模式必须有 reference_reports"

            report_inputs = self.tokenization(reference_reports, device=spatio_temp_feat.device,
                                              max_length=self.args['max_length'])
            decoder_input_ids, decoder_attention_mask, labels_ids = \
                self.obtain_decoder_input_ids(report_inputs)

            outputs = self.text_decoder.encoder_decoder(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                labels=labels_ids)
            loss_main = outputs['loss']

            # =================== CRM 辅助分类 loss ==========================
            loss_tc = loss_main.new_zeros(())
            if self.use_crm and change_labels is not None:
                if not isinstance(change_labels, torch.Tensor):
                    change_labels = torch.tensor(change_labels, dtype=torch.long, device=loss_main.device)
                else:
                    change_labels = change_labels.to(device=loss_main.device, dtype=torch.long)

                # 只有原始 label ∈ {1,2,3,4} 参与监督；0 (no_prior) 和 5 (none) 均跳过
                valid_mask = (change_labels >= 1) & (change_labels <= 4)

                # 类权重 (可选)
                w = None
                if self.change_class_weights is not None:
                    w = self.change_class_weights.to(
                        device=change_logits.device, dtype=change_logits.dtype)

                # 兼容: CRM 若返回 [B,M,K], 在 M 维 max-pool
                if change_logits.dim() == 3:
                    change_logits = change_logits.max(dim=1).values

                if valid_mask.any():
                    # label - 1: 原始 {1,2,3,4} → CRM 类别索引 {0,1,2,3}
                    tc_targets = change_labels[valid_mask] - 1   # shape: [N_valid]
                    loss_tc = F.cross_entropy(
                        change_logits[valid_mask],               # shape: [N_valid, 4]
                        tc_targets,                              # shape: [N_valid]，值域 {0,1,2,3}
                        weight=w)
                # else: loss_tc 保持为 0，不参与梯度

            total = loss_main + self.lambda_tc * loss_tc

            return {
                'loss':      total,
                'loss_main': loss_main.detach(),
                'loss_tc':   loss_tc.detach() if torch.is_tensor(loss_tc)
                             else torch.tensor(0.0, device=loss_main.device),
            }

        # =====================================================================
        # Val / Test branch
        # =====================================================================
        else:
            outputs = self.generate(encoder_outputs)
            generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in generated_reports]
            return generated_reports

    # ---------------------------------------------------------------------
    # steps
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        image_ids, reference_reports = batch['image_ids'], batch['reference_report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        change_labels = batch.get('change_label', None)

        if self.use_crm and self.current_epoch == 0 and batch_idx == 0:
            if change_labels is None:
                self.log_once(
                    "[tc][严重] batch 里没有 'change_label' 字段, tc_loss 将恒为 0! "
                    "请检查 FinetuneDataset/FinetuneCollateFn 是否把 change_label 放进 batch。")
            else:
                from collections import Counter
                _cl = change_labels.tolist() if torch.is_tensor(change_labels) else list(change_labels)
                _valid = sum(1 for x in _cl if 1 <= int(x) <= 4)
                self.log_once(
                    f"[tc] change_label 已进入 batch, 本 batch 分布={dict(Counter(int(x) for x in _cl))}, "
                    f"受监督(label∈1..4)样本数={_valid}/{len(_cl)}")

        out = self(current_study, context, reference_reports, prior_study, mode='train',
                   change_labels=change_labels)
        loss = out['loss']

        self.log_dict({
            'train_step_loss': loss.detach(),
            'train_step_main': out['loss_main'],
            'train_step_tc':   out['loss_tc'],
        }, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            self.log_once(
                f"Epoch {self.current_epoch}, step {batch_idx}/{self.trainer.num_training_batches}, "
                f"loss={loss.detach().cpu().item():.3f} "
                f"(main={out['loss_main'].item():.3f}, tc={out['loss_tc'].item():.3f}), "
                f"lr={self.optimizers().param_groups[0]['lr']}")

        self.train_loss_metric.update(loss.detach().cpu().item())
        self.train_loss_main_metric.update(out['loss_main'].detach().cpu().item())
        self.train_loss_tc_metric.update(out['loss_tc'].detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        image_ids, reference_reports = batch['image_ids'], batch['reference_report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        generated_reports = self(current_study, context, prior_study=prior_study, mode='val')
        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(reference_reports)

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        self.val_report_logger.update(generated_reports, dicom_ids=image_ids, reference_reports=reference_reports)
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def test_step(self, batch, batch_idx):
        image_ids, reference_reports = batch['image_ids'], batch['reference_report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        generated_reports = self(current_study, context, prior_study=prior_study, mode='test')
        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(reference_reports)

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")

        self.test_report_logger.update(generated_reports, dicom_ids=image_ids, reference_reports=reference_reports)
        self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)

    # ---------------------------------------------------------------------
    # epoch ends
    # ---------------------------------------------------------------------
    def on_train_epoch_end(self):
        el    = self.train_loss_metric.compute()
        emain = self.train_loss_main_metric.compute()
        etc   = self.train_loss_tc_metric.compute()
        for m in (self.train_loss_metric, self.train_loss_main_metric, self.train_loss_tc_metric):
            m.reset()

        spr_alpha_log = ""
        if self.soft_prior_registration is not None:
            try:
                a = torch.sigmoid(self.soft_prior_registration.cspa.alpha_raw).detach().cpu().item()
                spr_alpha_log = f"  spr_alpha(sigmoid)={a:.4f}"
                self.log_dict({'train_epoch_spr_alpha': a}, on_epoch=True, on_step=False, prog_bar=False)
            except Exception:
                pass

        self.log_dict({
            'train_epoch_loss':      el,
            'train_epoch_loss_main': emain,
            'train_epoch_loss_tc':   etc,
        }, on_epoch=True, on_step=False, prog_bar=False)
        self.log_once(
            "===============================================================\n"
            f"Epoch {self.current_epoch} TRAIN  loss={el:.4f}  "
            f"main={emain:.4f}  tc={etc:.4f}"
            f"{spr_alpha_log}  "
            f"lr={self.optimizers().param_groups[0]['lr']}"
            "\n===============================================================")

    def on_validation_epoch_end(self):
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()

        scores = {}
        output = self.val_f1chexbert_metrics.compute(); scores.update(output); self.val_f1chexbert_metrics.reset()
        output = self.val_coco_metrics.compute();       scores.update(output); self.val_coco_metrics.reset()

        scores['RB']  = scores.get('chen_bleu_4', 0.0)
        scores['RC']  = scores.get('chexbert_all_micro_f1', 0.0)
        scores['RCB'] = scores.get('chen_bleu_4', 0.0) + scores.get('chexbert_all_micro_f1', 0.0)
        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args['monitor_metric']] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {'best_epoch': self.current_epoch,
                                    'best_monitor_metric': scores[self.args['monitor_metric']]}

        key = (f"BLEU-1={scores.get('chen_bleu_1', 0):.4f}  "
               f"BLEU-4={scores.get('chen_bleu_4', 0):.4f}  "
               f"ROUGE-L={scores.get('chen_rouge_l', 0):.4f}  "
               f"METEOR={scores.get('chen_meteor', 0):.4f}  "
               f"CheXbert-F1={scores.get('chexbert_all_micro_f1', 0):.4f}  "
               f"RCB={scores.get('RCB', 0):.4f}")
        full = '\n'.join([f'    {k}: {v}' for k, v in scores.items()])
        self.log_once(
            "===============================================================\n"
            f"Epoch {self.current_epoch} VAL    {key}\n"
            f"  best so far: epoch={self.val_best_scores['best_epoch']}, "
            f"{self.args['monitor_metric']}={self.val_best_scores['best_monitor_metric']:.4f}\n"
            f"  full metrics:\n{full}"
            "\n===============================================================")

    def on_test_epoch_end(self):
        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_f1chexbert_metrics.compute(); scores.update(output); self.test_f1chexbert_metrics.reset()
        output = self.test_coco_metrics.compute();       scores.update(output); self.test_coco_metrics.reset()

        scores['RB']  = scores.get('chen_bleu_4', 0.0)
        scores['RC']  = scores.get('chexbert_all_micro_f1', 0.0)
        scores['RCB'] = scores.get('chen_bleu_4', 0.0) + scores.get('chexbert_all_micro_f1', 0.0)
        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        full = '\n'.join([f'    {k}: {v}' for k, v in scores.items()])
        self.log_once(
            "===============================================================\n"
            "TEST IS OVER. Final metrics:\n" + full +
            "\n===============================================================")


# =============================================================================
# Auxiliary networks (kept for back-compat with old checkpoints)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_result = torch.mean(x, dim=(2, 3), keepdim=True)
        max_result = torch.amax(x, dim=(2, 3), keepdim=True)
        return self.sigmoid(self.se(max_result) + self.se(avg_result))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result    = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_result, avg_result], 1)))


class CBAMBlock(nn.Module):
    def __init__(self, channel=12, reduction=2, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class LayerwiseFusion(nn.Module):
    def __init__(self, channel=12, reduction=4):
        super().__init__()
        self.cbam = CBAMBlock(channel=channel, reduction=reduction)
        self.projection = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, 1, kernel_size=1))
    def forward(self, x):
        x = self.cbam(x)
        x = self.projection(x)
        return x.squeeze(dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, 1, 0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, 1, 1, 0))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class VisualProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, 1, 0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, 1, 1, 0))
        self.norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)
