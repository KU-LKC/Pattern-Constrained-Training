"""
PatternLayout: a hardware spec compiled at a weight bit-width.

Weights ``[O, I, ...]`` are cut into tiles of ``tile_width`` consecutive input channels; each
tile is split into compressor segments in ``tile`` order, and every segment is projected with
element-wise free/tethered allocation. The ``I % tile_width`` remainder stays exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .allocation import SegmentAllocator
from .compiler import Certificate, compile_compressor
from .spec import HWSpec


@dataclass
class LayoutCertificate:
    segments: Dict[str, Certificate]

    @property
    def passed(self) -> bool:
        return all(c is not None and c.passed for c in self.segments.values())


class PatternLayout(nn.Module):
    def __init__(self, spec: HWSpec, bw: int):
        super().__init__()
        self.name = spec.name
        self.bw = int(bw)
        self.tile: Tuple[str, ...] = tuple(spec.tile)
        keys = list(dict.fromkeys(self.tile))
        self.segments = nn.ModuleDict({k: compile_compressor(spec.compressors[k], self.bw) for k in keys})
        offsets: Dict[str, List[int]] = {k: [] for k in keys}
        pos = 0
        for k in self.tile:
            offsets[k].append(pos)
            pos += int(self.segments[k].arity)
        self.tile_width = pos
        self.offsets = {k: tuple(v) for k, v in offsets.items()}
        self.allocators = {k: SegmentAllocator(self.segments[k]) for k in keys}
        self.certificate = LayoutCertificate({k: s.certificate for k, s in self.segments.items()})

    def project(self, q: torch.Tensor) -> torch.Tensor:
        """Project int weights ``[O, I, ...]`` tile by tile along I."""
        if q.ndim < 2:
            raise ValueError(f"expected weight rank >= 2, got {tuple(q.shape)}")
        O, I = int(q.shape[0]), int(q.shape[1])
        T = self.tile_width
        n = I // T
        if n == 0:
            return q
        end = n * T
        out = q.clone()
        tiles = out[:, :end].reshape(O, n, T, -1)
        for key, offs in self.offsets.items():
            comp = self.segments[key]
            A = int(comp.arity)
            seg = torch.stack([tiles[:, :, o:o + A] for o in offs], dim=2)        # [O, n, S, A, K]
            g = seg.movedim(3, -1).reshape(-1, A)
            pg = self.allocators[key].project(g, comp)
            pg = pg.reshape(O, n, len(offs), -1, A).movedim(-1, 3)
            for k, o in enumerate(offs):
                tiles[:, :, o:o + A] = pg[:, :, k]
        out[:, :end] = tiles.reshape(O, end, *q.shape[2:])
        return out

    def extra_repr(self) -> str:
        widths = "+".join(str(int(self.segments[k].arity)) for k in self.tile)
        return f"name={self.name}, tile={list(self.tile)} ({widths}={self.tile_width}), bw={self.bw}"


def compile_layout(spec: Any, bw: int) -> PatternLayout:
    """Hardware spec (path, dict or HWSpec) -> certified PatternLayout at weight bit-width ``bw``."""
    layout = PatternLayout(HWSpec.load(spec), bw)
    if not layout.certificate.passed:
        raise RuntimeError(f"hardware spec {layout.name!r} failed certification: {layout.certificate}")
    return layout
