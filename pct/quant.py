"""Symmetric fake quantization and the projected-weight straight-through estimator."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def weight_scale(w: torch.Tensor, w_bits: int) -> torch.Tensor:
    """Per-output-channel scale ``amax / (2^(b-1) - 1)`` shaped ``[O, 1, ...]``."""
    amax = w.detach().abs().reshape(w.shape[0], -1).amax(1)
    scale = (amax / ((1 << (int(w_bits) - 1)) - 1)).clamp(min=1e-8)
    return scale.view((-1,) + (1,) * (w.ndim - 1))


def quantize_symmetric_int(w: torch.Tensor, scale: torch.Tensor, w_bits: int) -> torch.Tensor:
    """Round-to-nearest symmetric quantization to signed int32."""
    hi = (1 << (int(w_bits) - 1)) - 1
    return torch.round((w / scale).clamp(min=-hi - 1, max=hi)).to(torch.int32)


class ProjectedWeightSTE(torch.autograd.Function):
    """quantize -> pattern projection -> dequantize; identity gradient to the float weight."""

    @staticmethod
    def forward(ctx, w, scale, w_bits, layout):
        q = quantize_symmetric_int(w, scale, w_bits)
        if layout is not None:
            q = layout.project(q)
        return (q.to(scale.dtype) * scale).to(w.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


def projected_weight(w: torch.Tensor, w_bits: int, layout: Optional[nn.Module]) -> torch.Tensor:
    """Fake-quantized weight; ``layout=None`` gives plain fake-quant STE (no projection)."""
    return ProjectedWeightSTE.apply(w, weight_scale(w, w_bits), int(w_bits), layout)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


class ActQuantizer(nn.Module):
    """Per-tensor symmetric activation fake quantizer; amax tracks an EMA of the batch max
    during training and is frozen at eval."""

    def __init__(self, a_bits: int = 8, ema_momentum: float = 0.99):
        super().__init__()
        self.a_bits = int(a_bits)
        self.ema_momentum = float(ema_momentum)
        self.register_buffer("amax", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _update_amax(self, x: torch.Tensor) -> None:
        cur = x.detach().abs().max()
        if not bool(self.initialized):
            self.amax.copy_(cur)
            self.initialized.fill_(True)
        else:
            m = self.ema_momentum
            self.amax.mul_(m).add_(cur * (1.0 - m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_amax(x)
        amax = self.amax if bool(self.initialized) else x.detach().abs().max()
        hi = (1 << (self.a_bits - 1)) - 1
        scale = (amax / hi).clamp(min=1e-8)
        return _ste_round((x / scale).clamp(min=-hi - 1, max=hi)) * scale

    def extra_repr(self) -> str:
        return f"a_bits={self.a_bits}, ema_momentum={self.ema_momentum}"
