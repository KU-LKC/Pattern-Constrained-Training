"""
Closed-form nearest projection of signed integers onto a forbidden-bit mask.

``project_to_mask(q, mask, B)`` returns, element-wise, the B-bit two's-complement value
closest to ``q`` whose bit pattern has no 1 on any bit set in ``mask``. Ties are broken
deterministically: smaller Hamming distance to ``q``'s pattern, then smaller magnitude,
then smaller unsigned pattern. The search runs in the offset-binary domain (u = q + 2^(B-1))
where signed order equals unsigned order; loops run only over the B bit positions.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _highest_set_bit(x: torch.Tensor, B: int) -> torch.Tensor:
    """Index of the highest set bit in [0, B); -1 where x == 0."""
    k = torch.full(x.shape, -1, dtype=torch.int64, device=x.device)
    for b in range(B):
        k = torch.where(((x >> b) & 1) != 0, torch.full_like(k, b), k)
    return k


def _lowest_set_bit(x: torch.Tensor, B: int) -> torch.Tensor:
    """Index of the lowest set bit in [0, B); -1 where x == 0."""
    k = torch.full(x.shape, -1, dtype=torch.int64, device=x.device)
    found = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    for b in range(B):
        take = (((x >> b) & 1) != 0) & ~found
        k = torch.where(take, torch.full_like(k, b), k)
        found = found | take
    return k


def _popcount(x: torch.Tensor, B: int) -> torch.Tensor:
    cnt = torch.zeros_like(x)
    for b in range(B):
        cnt = cnt + ((x >> b) & 1)
    return cnt


def _floor(u0: torch.Tensor, m: torch.Tensor, v: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Largest u <= u0 with u & m == v. Returns (u, valid)."""
    already = (u0 & m) == v
    k = _highest_set_bit((u0 & m) ^ v, B)

    bit_k = ((u0 >> k.clamp(min=0)) & 1) != 0
    case_a = (~already) & (k >= 0) & bit_k
    ones_k = torch.where(k >= 0, (torch.ones_like(u0) << k.clamp(min=0)) - 1, torch.zeros_like(u0))
    res_a = (u0 >> (k + 1) << (k + 1)) | ((~m | v) & ones_k)

    case_b = (~already) & (k >= 0) & (~bit_k)
    low = torch.where(k >= 0, (torch.ones_like(u0) << (k + 1).clamp(min=0)) - 1, torch.zeros_like(u0))
    j = _lowest_set_bit(u0 & (~m) & (~low), B)
    has_j = case_b & (j >= 0)
    ones_j = torch.where(j >= 0, (torch.ones_like(u0) << j.clamp(min=0)) - 1, torch.zeros_like(u0))
    res_b = (u0 >> (j + 1) << (j + 1)) | ((~m | v) & ones_j)

    result = torch.where(already, u0, torch.zeros_like(u0))
    result = torch.where(case_a, res_a, result)
    result = torch.where(has_j, res_b, result)
    return result, already | case_a | has_j


def _ceil(u0: torch.Tensor, m: torch.Tensor, v: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smallest u >= u0 with u & m == v. Returns (u, valid)."""
    already = (u0 & m) == v
    k = _highest_set_bit((u0 & m) ^ v, B)
    full = (1 << B) - 1

    bit_k = ((u0 >> k.clamp(min=0)) & 1) != 0
    case_a = (~already) & (k >= 0) & (~bit_k)
    ones_k = torch.where(k >= 0, (torch.ones_like(u0) << k.clamp(min=0)) - 1, torch.zeros_like(u0))
    res_a = (u0 >> (k + 1) << (k + 1)) | (torch.ones_like(u0) << k.clamp(min=0)) | (v & ones_k)
    res_a = torch.where(k >= 0, res_a, torch.zeros_like(u0))

    case_b = (~already) & (k >= 0) & bit_k
    low = torch.where(k >= 0, (torch.ones_like(u0) << (k + 1).clamp(min=0)) - 1, torch.zeros_like(u0))
    j = _lowest_set_bit((~u0) & (~m) & (~low) & full, B)
    has_j = case_b & (j >= 0)
    ones_j = torch.where(j >= 0, (torch.ones_like(u0) << j.clamp(min=0)) - 1, torch.zeros_like(u0))
    res_b = (u0 >> (j + 1) << (j + 1)) | (torch.ones_like(u0) << j.clamp(min=0)) | (v & ones_j)

    result = torch.where(already, u0, torch.zeros_like(u0))
    result = torch.where(case_a, res_a, result)
    result = torch.where(has_j, res_b, result)
    return result, already | case_a | has_j


def project_to_mask(q: torch.Tensor, mask: torch.Tensor, B: int) -> torch.Tensor:
    """Nearest B-bit signed value to ``q`` with ``pattern & mask == 0`` (integer tensors, broadcastable)."""
    B = int(B)
    H = 1 << (B - 1)
    full = (1 << B) - 1
    q64, m64 = torch.broadcast_tensors(q.to(torch.int64), mask.to(torch.int64) & full)
    u0 = q64 + H
    v = m64 & H                      # the sign bit is inverted in the offset-binary domain

    lo, lo_ok = _floor(u0, m64, v, B)
    hi, hi_ok = _ceil(u0, m64, v, B)

    dlo, dhi = u0 - lo, hi - u0
    q_pat, lo_pat, hi_pat = q64 & full, lo ^ H, hi ^ H
    hlo, hhi = _popcount(lo_pat ^ q_pat, B), _popcount(hi_pat ^ q_pat, B)
    slo, shi = (lo - H).abs(), (hi - H).abs()
    choose_lo = (dlo < dhi) | ((dlo == dhi) & (
        (hlo < hhi) | ((hlo == hhi) & ((slo < shi) | ((slo == shi) & (lo_pat <= hi_pat))))))

    out = torch.where(lo_ok & (choose_lo | ~hi_ok), lo, hi)
    out = torch.where(lo_ok | hi_ok, out, u0)
    return (out - H).to(q.dtype)
