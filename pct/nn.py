"""
Pattern-constrained quantized layers built from existing torch modules.

Weights are quantized per output channel, projected by a ``PatternLayout`` along the input
(reduction) axis, and dequantized; gradients pass straight through to the float weights.
Inputs are fake-quantized per tensor. With ``layout=None`` the layers are plain fake-quant
QAT layers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import ActQuantizer, projected_weight


class _PatternWeightMixin:
    weight: nn.Parameter
    w_bits: int
    layout: Optional[nn.Module]

    def projected_weight(self) -> torch.Tensor:
        # Recomputed every training forward; at eval under no_grad the weights are frozen,
        # so the result is cached until the parameter changes.
        if self.training or torch.is_grad_enabled():
            return projected_weight(self.weight, self.w_bits, self.layout)
        key = (self.weight._version, self.weight.data_ptr(), self.weight.device, self.weight.dtype)
        cache = self.__dict__.get("_wq_cache")
        if cache is None or cache[0] != key:
            cache = (key, projected_weight(self.weight, self.w_bits, self.layout))
            self.__dict__["_wq_cache"] = cache
        return cache[1]


class PatternConv2d(_PatternWeightMixin, nn.Module):
    def __init__(self, conv: nn.Conv2d, layout: Optional[nn.Module], *, w_bits: int, a_bits: int):
        super().__init__()
        if conv.padding_mode != "zeros":
            raise ValueError(f"padding_mode={conv.padding_mode!r} not supported")
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size, self.stride = conv.kernel_size, conv.stride
        self.padding, self.dilation, self.groups = conv.padding, conv.dilation, conv.groups
        self.w_bits, self.a_bits = int(w_bits), int(a_bits)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.input_quantizer = ActQuantizer(a_bits=self.a_bits)
        self.layout = layout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.input_quantizer(x), self.projected_weight(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, groups={self.groups}, w_bits={self.w_bits}, a_bits={self.a_bits}")


class PatternLinear(_PatternWeightMixin, nn.Module):
    """``weight`` is ``[out, in]``; ``from_conv1d`` handles HF ``Conv1D`` (``[in, out]``)."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], layout: Optional[nn.Module], *,
                 w_bits: int, a_bits: int):
        super().__init__()
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])
        self.w_bits, self.a_bits = int(w_bits), int(a_bits)
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.input_quantizer = ActQuantizer(a_bits=self.a_bits)
        self.layout = layout

    @classmethod
    def from_linear(cls, lin: nn.Linear, layout, **kw) -> "PatternLinear":
        return cls(lin.weight, lin.bias, layout, **kw)

    @classmethod
    def from_conv1d(cls, conv1d: nn.Module, layout, **kw) -> "PatternLinear":
        return cls(conv1d.weight.t(), conv1d.bias, layout, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.input_quantizer(x), self.projected_weight(), self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, w_bits={self.w_bits}, a_bits={self.a_bits}")


def _additive_mask(mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(mask, float("-inf"))
    return mask.to(dtype)


class PatternMultiheadAttention(nn.Module):
    """Drop-in for ``nn.MultiheadAttention`` whose q/k/v/out projections are PatternLinear.

    ``nn.MultiheadAttention`` passes ``in_proj_weight`` / ``out_proj.weight`` directly to
    ``F.multi_head_attention_forward``, so its projections are rebuilt explicitly here. Only
    weight x activation products are constrained; QK^T and attn x V stay exact.
    """

    def __init__(self, mha: nn.MultiheadAttention, layout, **kw):
        super().__init__()
        if mha.bias_k is not None or mha.add_zero_attn:
            raise ValueError("add_bias_kv / add_zero_attn not supported")
        E = mha.embed_dim
        self.embed_dim, self.num_heads, self.head_dim = E, mha.num_heads, E // mha.num_heads
        self.dropout, self.batch_first = float(mha.dropout), bool(mha.batch_first)
        if mha._qkv_same_embed_dim:
            wq, wk, wv = mha.in_proj_weight.chunk(3, dim=0)
        else:
            wq, wk, wv = mha.q_proj_weight, mha.k_proj_weight, mha.v_proj_weight
        bq = bk = bv = None
        if mha.in_proj_bias is not None:
            bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
        self.q_proj = PatternLinear(wq, bq, layout, **kw)
        self.k_proj = PatternLinear(wk, bk, layout, **kw)
        self.v_proj = PatternLinear(wv, bv, layout, **kw)
        self.out_proj = PatternLinear(mha.out_proj.weight, mha.out_proj.bias, layout, **kw)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        unbatched = query.dim() == 2
        if unbatched:
            query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        elif self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        L, N, _ = query.shape
        S = key.shape[0]
        H, hd = self.num_heads, self.head_dim

        def heads(t: torch.Tensor, n: int) -> torch.Tensor:
            return t.reshape(n, N, H, hd).permute(1, 2, 0, 3)  # [N, H, len, hd]

        q = heads(self.q_proj(query), L)
        k = heads(self.k_proj(key), S)
        v = heads(self.v_proj(value), S)

        mask = _additive_mask(attn_mask, q.dtype)
        if mask is not None:
            mask = mask.reshape(N, H, L, S) if mask.dim() == 3 else mask.view(1, 1, L, S)
        elif is_causal:
            mask = torch.full((L, S), float("-inf"), dtype=q.dtype, device=q.device).triu(1).view(1, 1, L, S)
        kpm = _additive_mask(key_padding_mask, q.dtype)
        if kpm is not None:
            kpm = kpm.view(N, 1, 1, S)
            mask = kpm if mask is None else mask + kpm
        p = self.dropout if self.training else 0.0

        weights = None
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            if mask is not None:
                scores = scores + mask
            attn = scores.softmax(-1)
            weights = attn.mean(1) if average_attn_weights else attn
            out = F.dropout(attn, p=p) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)

        out = self.out_proj(out.permute(2, 0, 1, 3).reshape(L, N, self.embed_dim))
        if unbatched:
            out = out.squeeze(1)
            weights = weights.squeeze(0) if weights is not None else None
        elif self.batch_first:
            out = out.transpose(0, 1)
        return out, weights


PATTERN_MODULES = (PatternConv2d, PatternLinear)
