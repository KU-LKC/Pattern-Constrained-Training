"""
Truth table -> certified error-free weight projection for one approximate compressor.

For an A-input compressor with truth table ``approx(pp)``:

  1. Error predicate      E(pp) = [approx(pp) != popcount(pp)], pp in {0,1}^A.
  2. Minimal patterns     M(E): submask-minimal supports of error-triggering inputs. Since the
                          activation bits are uncontrolled, an error is reachable on a bit plane
                          iff the weight bits of that plane cover some pattern in M(E).
  3. Roles                a minimum hitting set P of M(E) gives the *tethered* inputs; the rest
                          are *free*. Free weights are never changed. Each tethered weight is
                          projected, in order, to the nearest value whose bits avoid every plane
                          where (committed weights' bits + this input) could complete a pattern.
  4. Certificate          a bit-width independent per-plane proof (every free bit vector leaves
                          no reachable error), plus an exhaustive value-level check at small bw.

The forbidden-bit rule of each tethered step depends only on which inputs are committed, so it
is compiled once into a lookup table indexed by the committed inputs' bit signature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .projection import project_to_mask
from .spec import CompressorDef, TruthTable

ErrorFn = Callable[[Tuple[int, ...]], bool]
VALUE_CHECK_BW = 4
VALUE_CHECK_MAX_ARITY = 4


def error_fn_from_table(table: TruthTable) -> ErrorFn:
    tbl = {tuple(k): int(v) for k, v in table.items()}
    return lambda pp: tbl[tuple(pp)] != sum(pp)


def minimal_error_patterns(E: ErrorFn, A: int) -> List[frozenset]:
    """Supports of error inputs that contain no smaller error input."""
    minimal = set()
    for s in itertools.product((0, 1), repeat=A):
        if not E(s):
            continue
        supp = [i for i in range(A) if s[i]]
        if not any(E(tuple(1 if i in sub else 0 for i in range(A)))
                   for r in range(len(supp)) for sub in combinations(supp, r)):
            minimal.add(frozenset(supp))
    return list(minimal)


def minimum_hitting_sets(patterns: Sequence[frozenset], A: int) -> List[frozenset]:
    """All minimum-size index sets intersecting every pattern, in lexicographic order."""
    if not patterns:
        return [frozenset()]
    for r in range(1, A + 1):
        found = [frozenset(P) for P in combinations(range(A), r) if all(set(P) & p for p in patterns)]
        if found:
            return found
    return [frozenset(range(A))]


@dataclass(frozen=True)
class RoleAssignment:
    free: Tuple[int, ...]
    tethered: Tuple[int, ...]          # projection order


def candidate_assignments(patterns: Sequence[frozenset], A: int) -> List[RoleAssignment]:
    """Every combination of a minimum hitting set and a projection order."""
    out = []
    for P in minimum_hitting_sets(patterns, A):
        free = tuple(i for i in range(A) if i not in P)
        out.extend(RoleAssignment(free, order) for order in itertools.permutations(sorted(P)))
    return out


def default_assignment(patterns: Sequence[frozenset], A: int) -> RoleAssignment:
    """Lexicographically last minimum hitting set as tethered inputs (low indices stay free)."""
    P = sorted(minimum_hitting_sets(patterns, A)[-1])
    return RoleAssignment(free=tuple(i for i in range(A) if i not in P), tethered=tuple(P))


def per_plane_certify(free: Sequence[int], tethered: Sequence[int], E: ErrorFn, A: int) -> int:
    """Number of free bit vectors (one bit plane, any bw) for which the sequential projection
    still leaves an activation pattern that triggers E. Zero means error-free on every plane."""
    bad = 0
    for fbits in itertools.product((0, 1), repeat=len(free)):
        wbit = dict(zip(sorted(free), fbits))
        for pj in tethered:
            others = list(wbit)
            forced = False
            for act in itertools.product((0, 1), repeat=len(others)):
                bits = [0] * A
                for k, c in enumerate(others):
                    bits[c] = act[k] & wbit[c]
                bits[pj] = 1
                if E(tuple(bits)):
                    forced = True
                    break
            wbit[pj] = 0 if forced else 1
        if any(E(tuple(act[i] & wbit[i] for i in range(A))) for act in itertools.product((0, 1), repeat=A)):
            bad += 1
    return bad


def _error_table(E: ErrorFn, A: int) -> torch.Tensor:
    """Bool table of size 2^A; bit i of the index is input i."""
    return torch.tensor([bool(E(tuple((p >> i) & 1 for i in range(A)))) for p in range(1 << A)], dtype=torch.bool)


def _subset_or(t: torch.Tensor, k: int) -> torch.Tensor:
    """f[s] = OR_{u subset of s} t[u] over k-bit indices."""
    f = t.clone()
    idx = torch.arange(1 << k)
    for b in range(k):
        has = (idx & (1 << b)) != 0
        f = f.clone()
        f[has] = f[has] | f[idx[has] ^ (1 << b)]
    return f


def _forbidden_table(E_table: torch.Tensor, committed: Sequence[int], pj: int) -> torch.Tensor:
    """forbid[s] = 1 iff a 1-bit on input ``pj`` can trigger E given committed bit signature s
    (bit t of s is committed input ``committed[t]``), for some activation pattern."""
    k = len(committed)
    if k == 0:
        return E_table[1 << int(pj)].reshape(1)
    w = torch.arange(1 << k, dtype=torch.long)
    idx = torch.full((1 << k,), 1 << int(pj), dtype=torch.long)
    for t, i in enumerate(committed):
        idx = idx | (((w >> t) & 1) << int(i))
    return _subset_or(E_table[idx], k)


@dataclass
class Certificate:
    arity: int
    free: Tuple[int, ...]
    tethered: Tuple[int, ...]
    per_plane_errors: int
    value_check_bw: Optional[int]
    value_errors: Optional[int]

    @property
    def passed(self) -> bool:
        return self.per_plane_errors == 0 and not self.value_errors


class CompiledCompressor(nn.Module):
    """Role assignment + sequential projection for one compressor at weight bit-width ``bw``."""

    def __init__(self, cdef: CompressorDef, bw: int):
        super().__init__()
        self.name = cdef.name
        self.arity = A = int(cdef.arity)
        self.bw = int(bw)
        self.E = error_fn_from_table(cdef.table)
        if self.E(tuple([0] * A)):
            raise ValueError(f"{cdef.name}: error on the all-zero input cannot be removed by weight projection")
        self.minimal_patterns = minimal_error_patterns(self.E, A)
        roles = default_assignment(self.minimal_patterns, A)
        self.free, self.tethered = roles.free, roles.tethered

        E_table = _error_table(self.E, A)
        self.register_buffer("E_table", E_table, persistent=False)
        self._committed: List[Tuple[int, ...]] = []
        committed = list(self.free)
        for s, pj in enumerate(self.tethered):
            others = tuple(sorted(committed))
            self._committed.append(others)
            self.register_buffer(f"forbid_{s}", _forbidden_table(E_table, others, pj), persistent=False)
            committed.append(int(pj))
        self.certificate: Optional[Certificate] = None

    def project_group(self, g: torch.Tensor, bw: Optional[int] = None) -> torch.Tensor:
        """g: [..., arity] signed ints -> projected ints (free inputs unchanged)."""
        B = int(bw or self.bw)
        full = (1 << B) - 1
        out = g.clone()
        like = out[..., 0]
        for s, pj in enumerate(self.tethered):
            others = self._committed[s]
            forbid = getattr(self, f"forbid_{s}")
            pats = [out[..., i] & full for i in others]
            mask = torch.zeros_like(like)
            for b in range(B):
                if others:
                    sig = torch.zeros(like.shape, dtype=torch.long, device=like.device)
                    for t, p in enumerate(pats):
                        sig = sig | (((p >> b) & 1).to(torch.long) << t)
                    hit = forbid[sig]
                else:
                    hit = torch.full(like.shape, bool(forbid[0]), dtype=torch.bool, device=like.device)
                mask = mask | (hit.to(mask.dtype) << b)
            out[..., pj] = project_to_mask(out[..., pj], mask, B)
        return out

    @torch.no_grad()
    def certify(self, value_check_bw: Optional[int] = VALUE_CHECK_BW) -> Certificate:
        """Per-plane proof, plus (for arity <= 4) every value tuple at ``value_check_bw`` projected
        and checked for a reachable error on any bit plane."""
        A = self.arity
        per_plane = per_plane_certify(self.free, self.tethered, self.E, A)
        value_errors = None
        if value_check_bw is not None and A <= VALUE_CHECK_MAX_ARITY:
            B, dev = int(value_check_bw), self.E_table.device
            vals = torch.arange(-(1 << (B - 1)), 1 << (B - 1), dtype=torch.int32, device=dev)
            grid = torch.cartesian_prod(*([vals] * A)).reshape(-1, A)
            proj = self.project_group(grid, bw=B) & ((1 << B) - 1)
            reach = _subset_or(self.E_table.cpu(), A).to(dev)
            bad = torch.zeros(grid.shape[0], dtype=torch.bool, device=dev)
            for b in range(B):
                sig = sum(((proj[:, i] >> b) & 1).to(torch.long) << i for i in range(A))
                bad |= reach[sig]
            value_errors = int(bad.sum())
        self.certificate = Certificate(A, self.free, self.tethered, per_plane, value_check_bw, value_errors)
        return self.certificate

    def extra_repr(self) -> str:
        return f"name={self.name}, arity={self.arity}, bw={self.bw}, free={self.free}, tethered={self.tethered}"


def compile_compressor(cdef: CompressorDef, bw: int) -> CompiledCompressor:
    layer = CompiledCompressor(cdef, bw)
    cert = layer.certify()
    if not cert.passed:
        raise RuntimeError(f"compressor {cdef.name!r} failed certification: {cert}")
    return layer
