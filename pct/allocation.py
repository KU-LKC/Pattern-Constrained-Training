"""
Element-wise free/tethered allocation.

Every minimum hitting set of M(E) is a valid tethered set, so each tile (output channel,
tile index, kernel tap) and compressor segment may pick its own free set. The allocator
picks the free set with the largest sum |q|, so large weights stay free and small ones absorb
the projection.

Each candidate free set is the image of the compiled free set under an input permutation pi
that leaves M(E) invariant. A tile is permuted by its pi (gather), projected by the compiled
compressor, and permuted back (scatter). Legality is preserved because the certificate only
depends on M(E), which pi preserves. Candidates not reachable by such a permutation are
dropped. Ties keep the compiled default.

Selection rule per compressor:
  * arity 2 (e.g. OR):              the larger |q| is free
  * fully symmetric M(E) (e.g. 4:2): inputs ranked by |q| fill (free, tethered) in order
  * otherwise:                       argmax over candidates of sum |q| on the free inputs
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .compiler import CompiledCompressor, candidate_assignments


def _automorphisms(patterns: Sequence[frozenset], A: int) -> List[Tuple[int, ...]]:
    S = set(patterns)
    return [p for p in itertools.permutations(range(A)) if {frozenset(p[i] for i in s) for s in S} == S]


class SegmentAllocator:
    """Per-tile permutation (canonical slot -> physical input) for one compiled compressor."""

    def __init__(self, comp: CompiledCompressor):
        self.A = A = int(comp.arity)
        free = tuple(int(i) for i in comp.free)
        auts = _automorphisms(comp.minimal_patterns, A)
        frees: List[Tuple[int, ...]] = [free]
        for cand in candidate_assignments(comp.minimal_patterns, A):
            if tuple(sorted(cand.free)) not in [tuple(sorted(f)) for f in frees]:
                frees.append(tuple(cand.free))
        perms, kept = [], []
        for fr in frees:
            ok = [p for p in auts if {p[i] for i in free} == set(fr)]
            if ok:
                perms.append(ok[0])
                kept.append(fr)
        self.frees, self.perms = kept, perms
        self.slots = list(free) + [int(i) for i in comp.tethered]
        if len(kept) <= 1:
            self.mode = "fixed"
        elif A == 2:
            self.mode = "swap"
        elif len(auts) == len(list(itertools.permutations(range(A)))) and len(self.slots) == A:
            self.mode = "rank"
        else:
            self.mode = "argmax"
        self._cache: Dict[torch.device, Dict[str, torch.Tensor]] = {}

    def _tensors(self, device) -> Dict[str, torch.Tensor]:
        t = self._cache.get(device)
        if t is None:
            t = {
                "perms": torch.tensor(self.perms, dtype=torch.long, device=device),
                "slots": torch.tensor(self.slots, dtype=torch.long, device=device),
                "free_mask": torch.tensor([[1 if i in f else 0 for i in range(self.A)] for f in self.frees],
                                          dtype=torch.long, device=device),
            }
            self._cache[device] = t
        return t

    def perm(self, g: torch.Tensor) -> Optional[torch.Tensor]:
        """g: [N, A] ints -> [N, A] permutation (canonical slot -> physical input), or None for the default."""
        if self.mode == "fixed":
            return None
        t = self._tensors(g.device)
        a = g.abs()
        if self.mode == "swap":
            free, teth = self.slots
            flip = (a[:, teth] > a[:, free]).unsqueeze(1)
            base = t["perms"][0].unsqueeze(0)
            return torch.where(flip, base.flip(1), base)
        if self.mode == "rank":
            ranked = a.argsort(dim=1, descending=True, stable=True)
            return torch.empty_like(ranked).scatter_(1, t["slots"].expand_as(ranked), ranked)
        score = a.unsqueeze(1).mul(t["free_mask"].unsqueeze(0)).sum(-1)
        return t["perms"][score.argmax(1)]

    def project(self, g: torch.Tensor, comp: CompiledCompressor) -> torch.Tensor:
        """Element-wise allocated projection of ``g`` ([N, A] ints)."""
        p = self.perm(g)
        if p is None:
            return comp.project_group(g)
        return torch.empty_like(g).scatter_(1, p, comp.project_group(g.gather(1, p)))
