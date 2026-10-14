"""
Hardware specification: approximate-compressor truth tables and their tiling.

Compressor spec (JSON or YAML), one row per partial-product pattern:

    {"name": "or", "arity": 2, "approx_table": {"00": 0, "01": 1, "10": 1, "11": 1}}

Row key character ``i`` is the partial product ``pp_i`` of input ``i`` in the group
(``"bit_order": "pp0_first"``, default) or the reverse (``"pp0_last"``). A value is the
integer the compressor outputs; string values are read as binary unless ``output_weights``
gives per-output weights (e.g. ``[2, 2, 1]`` for Cout/C/S). An error is any row whose
output differs from the exact popcount.

Hardware spec (YAML or JSON) tiling compressors along the reduction axis:

    name: lsac_sxafa
    compressors:
      afa: specs/lsac_sxafa.json      # spec path or inline {arity, approx_table, ...}
      or: specs/or.json
    tile: [afa, afa, or]              # tile width = sum of arities (3 + 3 + 2 = 8)

A compressor spec file is also accepted as a one-compressor hardware spec. Input channels
beyond the last full tile are left exact.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BIT_ORDERS = ("pp0_first", "pp0_last")

TruthTable = Dict[Tuple[int, ...], int]


@dataclass
class CompressorDef:
    name: str
    arity: int
    table: TruthTable


def parse_truth_table(
    raw: Mapping[Any, Any],
    arity: int,
    bit_order: str = "pp0_first",
    output_weights: Optional[Sequence[int]] = None,
) -> TruthTable:
    """Rows ``{"0101": 2 | "10", ...}`` -> ``{(pp0, ..., pp_{A-1}): value}`` (all 2^A rows required)."""
    if bit_order not in BIT_ORDERS:
        raise ValueError(f"bit_order must be one of {BIT_ORDERS}, got {bit_order!r}")
    A = int(arity)
    out: TruthTable = {}
    for key, value in raw.items():
        bits = tuple(int(b) for b in key) if isinstance(key, (str, tuple, list)) else None
        if bits is None or len(bits) != A or any(b not in (0, 1) for b in bits):
            raise ValueError(f"row key {key!r} is not a {A}-bit pattern")
        if bit_order == "pp0_last":
            bits = bits[::-1]
        if isinstance(value, str):
            outs = [int(c) for c in value]
            if output_weights is not None:
                if len(output_weights) != len(outs):
                    raise ValueError(f"row {key!r}: {len(outs)} outputs vs output_weights {output_weights}")
                value = sum(int(w) * o for w, o in zip(output_weights, outs))
            else:
                value = int(value, 2)
        out[bits] = int(value)
    missing = [p for p in itertools.product((0, 1), repeat=A) if p not in out]
    if missing:
        raise ValueError(f"truth table incomplete: {len(missing)} of {1 << A} rows missing (e.g. {missing[0]})")
    return out


def _read(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def _resolve_path(src: Union[str, Path], base_dir: Optional[Path]) -> Path:
    p = Path(src)
    candidates = [p]
    if not p.is_absolute():
        if base_dir is not None:
            candidates.append(base_dir / p)
        candidates.append(REPO_ROOT / p)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"spec file {str(src)!r} not found")


def _compressor_from_dict(data: Mapping[str, Any], default_name: str) -> CompressorDef:
    if "approx_table" not in data:
        raise ValueError(f"compressor {default_name!r} needs an 'approx_table'")
    raw = data["approx_table"]
    arity = int(data.get("arity") or len(next(iter(raw.keys()))))
    table = parse_truth_table(raw, arity, data.get("bit_order", "pp0_first"), data.get("output_weights"))
    return CompressorDef(name=str(data.get("name", default_name)), arity=arity, table=table)


def load_compressor(entry: Union[str, Path, Mapping[str, Any]], base_dir: Optional[Path] = None) -> CompressorDef:
    """Compressor spec path or inline ``{arity, approx_table, bit_order, output_weights, name}``."""
    if isinstance(entry, Mapping):
        return _compressor_from_dict(entry, "inline")
    path = _resolve_path(entry, base_dir)
    return _compressor_from_dict(_read(path), path.stem)


@dataclass
class HWSpec:
    name: str
    compressors: Dict[str, CompressorDef]
    tile: List[str]

    def __post_init__(self) -> None:
        if not self.tile:
            raise ValueError("tile must list at least one compressor")
        unknown = [k for k in self.tile if k not in self.compressors]
        if unknown:
            raise ValueError(f"tile references undefined compressors: {unknown}")

    @property
    def tile_width(self) -> int:
        return sum(self.compressors[k].arity for k in self.tile)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base_dir: Optional[Path] = None) -> "HWSpec":
        if "compressors" not in data:
            cdef = _compressor_from_dict(data, "compressor")
            return cls(name=cdef.name, compressors={cdef.name: cdef}, tile=[cdef.name])
        comps = {str(k): load_compressor(v, base_dir) for k, v in data["compressors"].items()}
        tile = [str(t) for t in (data.get("tile") or list(comps)[:1])]
        return cls(name=str(data.get("name", "_".join(tile))), compressors=comps, tile=tile)

    @classmethod
    def load(cls, src: Union[str, Path, Mapping[str, Any], "HWSpec"]) -> "HWSpec":
        """Hardware or compressor spec file (.json / .yaml), dict, or HWSpec."""
        if isinstance(src, HWSpec):
            return src
        if isinstance(src, Mapping):
            return cls.from_dict(src)
        path = _resolve_path(src, None)
        return cls.from_dict(_read(path), base_dir=path.parent)

    def to_dict(self) -> Dict[str, Any]:
        """Self-contained form accepted by ``from_dict`` (inline pp0_first tables)."""
        comps = {
            k: {"name": c.name, "arity": c.arity,
                "approx_table": {"".join(map(str, pp)): int(v) for pp, v in sorted(c.table.items())}}
            for k, c in self.compressors.items()
        }
        return {"name": self.name, "compressors": comps, "tile": list(self.tile)}
