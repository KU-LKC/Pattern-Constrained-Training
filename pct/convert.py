"""
Convert a torch model's Conv2d / Linear (+ HF Conv1D, nn.MultiheadAttention) into
pattern-constrained quantized layers driven by a compiled ``PatternLayout``.

    layout = compile_layout("configs/hw/lsac_sxafa.yaml", bw=8)
    report = convert_model(model, layout, w_bits=8, a_bits=8)
    print(report.summary())

By default the first and last MAC layers (stem conv / patch embedding and classifier head)
stay in floating point. Layers whose reduction width per group is smaller than the tile
(e.g. depthwise conv, 3-channel stem) are skipped since no tile could be formed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

import torch.nn as nn

from .layout import PatternLayout
from .nn import PATTERN_MODULES, PatternConv2d, PatternLinear, PatternMultiheadAttention

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None

_UNSUPPORTED = (nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


@dataclass
class ConvertPolicy:
    include: Sequence[str] = ()          # regexes on qualified names; empty = everything
    exclude: Sequence[str] = ()
    keep_first: bool = True              # first MAC layer in module order stays float
    keep_last: bool = True               # last MAC layer (classifier head) stays float


@dataclass
class LayerRecord:
    name: str
    kind: str
    reduction: int                       # input width per output (in_features or in_channels / groups)
    status: str                          # converted | skipped
    reason: str = ""
    exact_frac: float = 0.0              # fraction of reduction channels left exact (remainder)


@dataclass
class ConvertReport:
    tile: int
    records: List[LayerRecord] = field(default_factory=list)

    @property
    def converted(self) -> List[LayerRecord]:
        return [r for r in self.records if r.status == "converted"]

    @property
    def skipped(self) -> List[LayerRecord]:
        return [r for r in self.records if r.status == "skipped"]

    def summary(self) -> str:
        lines = [f"converted {len(self.converted)} / {len(self.records)} MAC layers (tile={self.tile})"]
        for r in self.records:
            extra = f" exact_remainder={r.exact_frac:.1%}" if r.status == "converted" and r.exact_frac else ""
            why = f" ({r.reason})" if r.reason else ""
            lines.append(f"  [{r.status:9s}] {r.name:45s} {r.kind:18s} K={r.reduction}{extra}{why}")
        return "\n".join(lines)


def _kind(m: nn.Module) -> Optional[str]:
    if isinstance(m, nn.MultiheadAttention):
        return "MultiheadAttention"
    if isinstance(m, nn.Conv2d):
        return "Conv2d"
    if isinstance(m, nn.Linear):
        return "Linear"
    if HFConv1D is not None and isinstance(m, HFConv1D):
        return "HFConv1D"
    if isinstance(m, _UNSUPPORTED):
        return type(m).__name__
    return None


def _reduction(m: nn.Module, kind: str) -> int:
    if kind == "Conv2d":
        return m.in_channels // m.groups
    if kind == "Linear":
        return m.in_features
    if kind == "HFConv1D":
        return int(m.weight.shape[0])
    if kind == "MultiheadAttention":
        return m.embed_dim
    return int(m.weight.shape[1]) if hasattr(m, "weight") else 0


def _collect(model: nn.Module) -> List[Tuple[str, nn.Module, str]]:
    """MAC modules in registration order, not descending into matched modules."""
    found: List[Tuple[str, nn.Module, str]] = []

    def walk(mod: nn.Module, prefix: str) -> None:
        for name, child in mod.named_children():
            q = f"{prefix}.{name}" if prefix else name
            kind = _kind(child)
            if kind is not None:
                found.append((q, child, kind))
            elif not isinstance(child, PATTERN_MODULES + (PatternMultiheadAttention,)):
                walk(child, q)

    walk(model, "")
    return found


def _set_module(root: nn.Module, qualified: str, new: nn.Module) -> None:
    parent = root
    *path, leaf = qualified.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def _build(m: nn.Module, kind: str, layout, kw: Mapping) -> nn.Module:
    if kind == "Conv2d":
        new = PatternConv2d(m, layout, **kw)
    elif kind == "Linear":
        new = PatternLinear.from_linear(m, layout, **kw)
    elif kind == "HFConv1D":
        new = PatternLinear.from_conv1d(m, layout, **kw)
    else:
        new = PatternMultiheadAttention(m, layout, **kw)
    ref = next(m.parameters())
    return new.to(device=ref.device, dtype=ref.dtype)


def convert_model(
    model: nn.Module,
    layout: PatternLayout,
    *,
    w_bits: Optional[int] = None,
    a_bits: int = 8,
    project: bool = True,
    policy: Optional[ConvertPolicy] = None,
) -> ConvertReport:
    """In-place conversion; returns a per-layer report. ``project=False`` converts to plain
    fake-quant layers (same layer selection, no pattern projection)."""
    policy = policy or ConvertPolicy()
    w_bits = int(w_bits if w_bits is not None else layout.bw)
    if w_bits != int(layout.bw):
        raise ValueError(f"w_bits={w_bits} must match layout.bw={layout.bw} (compile the layout at this bw)")
    tile = int(layout.tile_width)
    kw = dict(w_bits=w_bits, a_bits=a_bits)
    inc = [re.compile(p) for p in policy.include]
    exc = [re.compile(p) for p in policy.exclude]

    targets = _collect(model)
    mac_idx = [i for i, (_, m, _) in enumerate(targets) if not isinstance(m, _UNSUPPORTED)]
    first = mac_idx[0] if mac_idx else None
    last = mac_idx[-1] if mac_idx else None

    report = ConvertReport(tile=tile)
    for i, (name, mod, kind) in enumerate(targets):
        K = _reduction(mod, kind)
        rec = LayerRecord(name=name, kind=kind, reduction=K, status="skipped")
        if isinstance(mod, _UNSUPPORTED):
            rec.reason = "unsupported layer type"
        elif policy.keep_first and i == first:
            rec.reason = "keep_first (float)"
        elif policy.keep_last and i == last:
            rec.reason = "keep_last (float)"
        elif inc and not any(p.search(name) for p in inc):
            rec.reason = "not in include"
        elif any(p.search(name) for p in exc):
            rec.reason = "excluded"
        elif K < tile:
            rec.reason = f"reduction {K} < tile {tile}"
        else:
            try:
                _set_module(model, name, _build(mod, kind, layout if project else None, kw))
            except ValueError as e:
                rec.reason = str(e)
            else:
                rec.status = "converted"
                rec.exact_frac = (K % tile) / K
        report.records.append(rec)
    return report


def is_pattern_module(m: nn.Module) -> bool:
    return isinstance(m, PATTERN_MODULES)
