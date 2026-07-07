"""
Soft Prior Registration (SPR) Module
=====================================

放在 CRM 之前，将既往影像 token 软对齐到当前影像 token，并显式构造变化表示。

整体由两部分组成：

    SPR = CSPA + ECE

CSPA  (Current-guided Soft Prior Alignment)
-------------------------------------------
以当前影像 token 作为 Query，既往影像 token 作为 Key/Value，
通过 cross-attention 得到对齐到当前影像位置的 V_pri_aligned。

    Q_cur         = W_q ( LN(V_cur) )
    K_pri         = W_k ( LN(V_pri) )
    V_pri_value   = W_v ( V_pri )                          # 不过 LN
    A             = softmax( Q_cur K_pri^T / sqrt(d_head) ) # [B, H, S, S]
    V_pri_soft    = W_out ( A · V_pri_value )
    alpha         = sigmoid(alpha_raw)                     # α ∈ (0, 1)
    V_pri_aligned = V_pri + alpha * (V_pri_soft - V_pri)
                  = (1 - alpha) * V_pri + alpha * V_pri_soft

α 初始化为 0.1，意味着训练开始时 V_pri_aligned ≈ V_pri，
不至于一开始就剧烈改变 prior 的分布。

ECE  (Explicit Change Encoder)
------------------------------
根据 V_cur 和 V_pri_aligned 构造显式变化表示 V_delta：

    signed_delta  = V_cur - V_pri_aligned
    abs_delta     = |signed_delta|
    interaction   = V_cur * V_pri_aligned
    V_delta       = MLP( [signed_delta, abs_delta, interaction] )

prior_mask = 0 的样本，V_pri_aligned 和 V_delta 都会被乘以 0 屏蔽掉。

V_cur 不被修改 (V_cur_cmp == V_cur)，避免给 CRM 的当前路径引入额外噪声。
"""

import torch
import torch.nn as nn


# =============================================================================
# CSPA
# =============================================================================
class CSPA(nn.Module):
    """Current-guided Soft Prior Alignment via multi-head cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        init_alpha: float = 0.1,
    ):
        super().__init__()

        assert dim % num_heads == 0, \
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Shared norm: current 和 prior 共享同一个 LayerNorm
        # 只对 Q / K 的输入做归一化，V 使用原始 V_pri 投影
        self.shared_ln = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Learnable residual gate:
        # alpha = sigmoid(alpha_raw), alpha ∈ (0, 1)
        # init_alpha 是初始对齐强度，不是固定值
        init_alpha = max(min(float(init_alpha), 1.0 - 1e-4), 1e-4)
        self.alpha_raw = nn.Parameter(
            torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
        )

    def forward(
        self,
        V_cur: torch.Tensor,
        V_pri: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        V_cur, V_pri: [B, S, D]

        Returns
        -------
        V_pri_aligned : [B, S, D]
            Prior feature softly aligned to current feature.

        alpha : tensor
            Learnable residual alignment coefficient.

        attn : [B, H, S, S] or None
            Cross-attention weights for visualization.
        """
        assert V_cur.shape == V_pri.shape, \
            f"V_cur shape {V_cur.shape} must match V_pri shape {V_pri.shape}"

        B, S, D = V_cur.shape
        H, Dh = self.num_heads, self.head_dim

        # 1. Shared norm for Q/K inputs
        V_cur_n = self.shared_ln(V_cur)
        V_pri_n = self.shared_ln(V_pri)

        # 2. Q from normed current; K from normed prior; V from raw prior
        Q = self.q_proj(V_cur_n).view(B, S, H, Dh).transpose(1, 2)  # [B, H, S, Dh]
        K = self.k_proj(V_pri_n).view(B, S, H, Dh).transpose(1, 2)  # [B, H, S, Dh]
        V = self.v_proj(V_pri).view(B, S, H, Dh).transpose(1, 2)    # [B, H, S, Dh]

        # 3. Current-guided cross-attention
        scale = Dh ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale                    # [B, H, S, S]
        attn = attn.softmax(dim=-1)

        attn_d = self.attn_dropout(attn)

        # 4. Current-guided prior feature
        V_pri_soft = (attn_d @ V).transpose(1, 2).contiguous().view(B, S, D)
        V_pri_soft = self.out_proj(V_pri_soft)

        # 5. Learnable soft residual fusion
        alpha = torch.sigmoid(self.alpha_raw)

        # Equivalent to:
        # V_pri_aligned = V_pri + alpha * (V_pri_soft - V_pri)
        V_pri_aligned = (1.0 - alpha) * V_pri + alpha * V_pri_soft

        if return_attn:
            return V_pri_aligned, alpha, attn

        return V_pri_aligned, alpha, None



# =============================================================================
# ECE
# =============================================================================
class ECE(nn.Module):
    """
    Explicit Change Encoder.

    Builds V_delta by concatenating three views (signed diff, absolute diff,
    and element-wise interaction) of the (current, aligned-prior) pair,
    then passing through a small MLP.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # 输入拼成 3D，先 norm 再过 MLP，训练稳定
        self.change_pre_norm = nn.LayerNorm(3 * dim)
        self.change_mlp = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, V_cur_cmp: torch.Tensor,
                V_pri_aligned: torch.Tensor) -> torch.Tensor:
        signed_delta = V_cur_cmp - V_pri_aligned
        abs_delta    = signed_delta.abs()
        interaction  = V_cur_cmp * V_pri_aligned

        feats = torch.cat([
            signed_delta,
            abs_delta,
            interaction,
        ], dim=-1)                                # [B, S, 3D]

        feats = self.change_pre_norm(feats)
        V_delta = self.change_mlp(feats)          # [B, S, D]
        return V_delta


# =============================================================================
# SPR  (top-level wrapper)
# =============================================================================
class SoftPriorRegistration(nn.Module):
    """
    Inputs
    ------
    V_cur       : [B, S, D]     current image tokens (pure, without temporal pos)
    V_pri       : [B, S, D]     prior image tokens (zeros where prior_mask = 0)
    prior_mask  : [B]           1.0 if prior exists, 0.0 otherwise

    Outputs
    -------
    V_cur_cmp     : [B, S, D]   == V_cur (not modified)
    V_pri_aligned : [B, S, D]   prior softly aligned to current
    V_delta       : [B, S, D]   explicit change features from ECE
    aux           : dict        {'alpha': tensor, optional 'attn': tensor}
    """

    def __init__(self, dim: int = 768, num_heads: int = 8,
                 dropout: float = 0.1, init_alpha: float = 0.1):
        super().__init__()
        self.dim = dim
        self.cspa = CSPA(
            dim=dim, num_heads=num_heads,
            dropout=dropout, init_alpha=init_alpha,
        )
        self.ece = ECE(dim=dim, dropout=dropout)

    def forward(self, V_cur: torch.Tensor, V_pri: torch.Tensor,
                prior_mask: torch.Tensor = None,
                return_attn: bool = False):
        # 1. CSPA: soft-align V_pri to V_cur
        V_pri_aligned, alpha, attn = self.cspa(V_cur, V_pri, return_attn=return_attn)

        # 2. V_cur_cmp is V_cur unchanged (保持 CRM 当前路径分布稳定)
        V_cur_cmp = V_cur

        # 3. ECE: build explicit change features
        V_delta = self.ece(V_cur_cmp, V_pri_aligned)

        # 4. Apply prior mask: 没有 prior 的样本，V_pri_aligned 和 V_delta 置零
        if prior_mask is not None:
            mask = prior_mask.to(dtype=V_cur.dtype).view(-1, 1, 1)
            V_pri_aligned = V_pri_aligned * mask
            V_delta       = V_delta       * mask

        aux = {'alpha': alpha.detach()}
        if return_attn and attn is not None:
            aux['attn'] = attn.detach()

        return V_cur_cmp, V_pri_aligned, V_delta, aux

