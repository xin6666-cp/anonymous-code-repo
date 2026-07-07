"""
Comparison Reasoning Module (CRM)
=================================

Explicitly models the visual change between current and prior chest X-ray
features. Produces a set of comparison query latents C that are fed
directly into the report decoder as fine-grained "change-aware" evidence
(no clinical-context gating is applied to C).

★ 本版本新增：支持外部传入 V_delta（来自 SPR/ECE）。

    raw_delta = LN_cur(V_cur) - LN_pri(V_pri)
    if V_delta is not None:
        Delta = LN_delta(V_delta)      # 用外部显式变化表示替换 raw_delta
    else:
        Delta = raw_delta

随后走 Perceiver-style 抽取，直接输出 C（不再做临床上下文 gating）。

Inputs:
    V_cur      : [B, s, d]  current image tokens (SPR 后即 V_cur_cmp == V_cur)
    V_pri      : [B, s, d]  prior image tokens (SPR 后即 V_pri_aligned)
    prior_mask : [B]        1.0 if prior exists, 0.0 otherwise
    V_delta    : [B, s, d]  explicit change features from SPR/ECE (optional)

Outputs:
    C             : [B, M, d]
    change_logits : [B, K]  per-sample change logits (max-pooled over M)
"""

import torch
import torch.nn as nn


class PerceiverBlock(nn.Module):
    """Pre-norm cross-attention + FFN."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_key_padding_mask=None) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_n, kv_n, kv_n, key_padding_mask=kv_key_padding_mask)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class ComparisonReasoningModule(nn.Module):
    """
    Args:
        dim:                    hidden dimension (typically 768).
        num_comparison_queries: M, number of learnable comparison query tokens.
        num_perceiver_layers:   number of cross-attention blocks.
        num_heads:              attention heads.
        num_change_classes:     output classes of the auxiliary classifier.
        dropout:                dropout rate.
    """

    def __init__(
        self,
        dim: int = 768,
        num_comparison_queries: int = 16,
        num_perceiver_layers: int = 2,
        num_heads: int = 8,
        num_change_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.M = num_comparison_queries
        self.num_change_classes = num_change_classes

        # Learnable comparison query tokens
        self.comparison_queries = nn.Parameter(torch.randn(1, num_comparison_queries, dim) * 0.02)

        # Pre-norms for the three views of the change-evidence pool
        self.ln_cur = nn.LayerNorm(dim)
        self.ln_pri = nn.LayerNorm(dim)
        # ★ NEW: norm for the externally provided V_delta (from SPR/ECE)
        self.ln_delta = nn.LayerNorm(dim)

        # Token-type embeddings: 0=cur, 1=pri, 2=delta
        # ★ 注意：type_embedding 是在 SPR 之后、CRM attention 之前加的
        self.type_embedding = nn.Embedding(3, dim)
        nn.init.normal_(self.type_embedding.weight, std=0.02)

        # Perceiver-style extraction
        self.perceiver_blocks = nn.ModuleList(
            [PerceiverBlock(dim, num_heads, dropout=dropout)
             for _ in range(num_perceiver_layers)]
        )

        # Auxiliary change classifier (per-query, then max-pool over M)
        self.change_classifier = nn.Linear(dim, num_change_classes)

    def forward(
        self,
        V_cur: torch.Tensor,
        V_pri: torch.Tensor,
        prior_mask: torch.Tensor = None,
        V_delta: torch.Tensor = None,
    ):
        """
        Returns
        -------
        C : [B, M, d]
        change_logits : [B, K]
        """
        B, s, d = V_cur.shape
        device = V_cur.device

        # --- Step 1. Build the three views of the change-evidence pool ---
        V_cur_n = self.ln_cur(V_cur)
        V_pri_n = self.ln_pri(V_pri)
        raw_delta = V_cur_n - V_pri_n

        if V_delta is None:
            Delta = raw_delta
        else:
            # 有外部 V_delta 时，直接以其归一化结果作为 Δ（替换 raw_delta；
            # 此分支下上面算出的 raw_delta 不再参与后续计算）
            Delta = self.ln_delta(V_delta)

        # 没有 prior 的样本，把 prior 和 delta 视作 0（attention 端用 padding mask 也屏蔽）
        if prior_mask is not None:
            mask = prior_mask.to(dtype=V_cur.dtype).view(B, 1, 1)
            V_pri_n = V_pri_n * mask
            Delta   = Delta   * mask

        # Add type embeddings AFTER SPR (cur/pri/delta 三路)
        type_ids_cur = torch.zeros(s, dtype=torch.long, device=device)
        type_ids_pri = torch.ones(s,  dtype=torch.long, device=device)
        type_ids_del = torch.full((s,), 2, dtype=torch.long, device=device)
        type_ids = torch.cat([type_ids_cur, type_ids_pri, type_ids_del], dim=0)   # [3s]
        type_emb = self.type_embedding(type_ids).unsqueeze(0)                     # [1,3s,d]

        V_ct = torch.cat([V_cur_n, V_pri_n, Delta], dim=1) + type_emb             # [B,3s,d]

        # Key-padding mask: hide prior / delta tokens when no prior available
        kv_key_padding_mask = None
        if prior_mask is not None:
            pad = (prior_mask < 0.5).view(B, 1)                                   # [B,1]
            pad_cur = torch.zeros(B, s, dtype=torch.bool, device=device)
            pad_pri = pad.expand(B, s)
            pad_del = pad.expand(B, s)
            kv_key_padding_mask = torch.cat([pad_cur, pad_pri, pad_del], dim=1)   # [B,3s]

        # --- Step 2. Perceiver extraction ---
        C = self.comparison_queries.expand(B, -1, -1).contiguous()                # [B,M,d]
        for block in self.perceiver_blocks:
            C = block(C, V_ct, kv_key_padding_mask=kv_key_padding_mask)

        # --- Step 3. Auxiliary change classification ---
        # 注：C 不再经临床上下文 gating，Perceiver 抽取后直接输出，
        #     由外部与 context / spatio-temporal latents 拼接后送入解码器。
        # per-query → [B,M,K]，在 M 维 max-pool → [B,K]
        change_logits_all = self.change_classifier(C)
        change_logits = change_logits_all.max(dim=1).values

        return C, change_logits

