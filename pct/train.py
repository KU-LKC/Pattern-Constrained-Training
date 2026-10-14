"""
Command-line entry point. Every ``TrainConfig`` field is a flag; ``--config`` YAML supplies
defaults and flags override them.

  # QAT baseline (no projection), then PCT fine-tuning with the OR compressor
  python -m pct.train --config configs/exp/cifar10_resnet20_qat.yaml
  python -m pct.train --config configs/exp/cifar10_resnet20_pct.yaml --hw configs/hw/or.yaml

  # ImageNet ResNet-18 from torchvision weights, LSAC tile, 4 GPUs
  python -m pct.train --config configs/exp/imagenet_resnet18_pct.yaml --hw configs/hw/lsac_sxafa.yaml --gpus 0,1,2,3

  # compile the spec, convert the model, print roles and the conversion report
  python -m pct.train --model timm:vit_base_patch16_224 --dataset imagenet1k --hw specs/or.json --dry_run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .trainer import TrainConfig


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if s.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def _field_parser(f: dataclasses.Field):
    hint = typing.get_type_hints(TrainConfig)[f.name]
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union and type(None) in args:
        hint = next(a for a in args if a is not type(None))
        origin = typing.get_origin(hint)
    if hint is bool:
        return dict(type=_bool, nargs="?", const=True)
    if origin in (list, List):
        return dict(type=str, nargs="*")
    if origin in (dict, Dict) or hint is Any:
        return dict(type=lambda s: yaml.safe_load(s))
    return dict(type=hint)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pattern-constrained training (PCT) for approximate compute-in-memory",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", default="", help="YAML with TrainConfig fields")
    p.add_argument("--dry_run", action="store_true", help="build and convert only, print the report")
    for f in dataclasses.fields(TrainConfig):
        p.add_argument(f"--{f.name}", default=argparse.SUPPRESS, **_field_parser(f))
    return p


def config_from_args(argv: Optional[Sequence[str]] = None):
    args = vars(build_parser().parse_args(argv))
    config_path, dry_run = args.pop("config"), args.pop("dry_run")
    base: Dict[str, Any] = {}
    if config_path:
        with open(config_path) as fh:
            base = yaml.safe_load(fh) or {}
    base.update(args)
    return TrainConfig.from_dict(base), dry_run


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg, dry_run = config_from_args(argv)
    if dry_run:
        import torch
        from .trainer import build
        _, layout, report, dc = build(cfg, torch.device("cpu"))
        for k, s in layout.segments.items():
            print(f"{k}: arity={s.arity} free={s.free} tethered={s.tethered} certificate={s.certificate}")
        print(f"tile={list(layout.tile)} width={layout.tile_width} bw={layout.bw} data={dc}")
        print(report.summary())
        return
    from .trainer import run
    res = run(cfg)
    if res is not None:
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
