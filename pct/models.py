"""
Model registry: ``<source>:<name>`` -> float ``nn.Module`` + the preprocessing it expects.
Conversion to pattern layers happens afterwards (``pct.convert.convert_model``).

  builtin:resnet20                 CIFAR ResNet-20
  builtin:resnet18 / resnet34      ResNet-18/34 with a 3x3 stem and no max-pool (32/64 px inputs)
  torchvision:resnet18             any torchvision classification model (resnet50, vit_b_16, ...)
  timm:vit_base_patch16_224        any timm model, incl. ``timm:hf-hub:<repo>``
  hf:google/vit-base-patch16-224   any HF AutoModelForImageClassification

With ``pretrained`` the model's own data config (size / mean / std / interpolation / crop) is
returned; otherwise the dataset defaults apply unless ``img_size`` is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F

from .data import DataConfig


# ----------------------------------------------------------------------------- built-in ResNets

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.residual_function(x) + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    """ResNet-18/34 for small images: 3x3 stride-1 stem, no max-pool."""

    def __init__(self, num_block, num_classes=10, width=64):
        super().__init__()
        self.in_channels = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.conv2_x = self._make_layer(width, num_block[0], 1)
        self.conv3_x = self._make_layer(width * 2, num_block[1], 2)
        self.conv4_x = self._make_layer(width * 4, num_block[2], 2)
        self.conv5_x = self._make_layer(width * 8, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv5_x(self.conv4_x(self.conv3_x(self.conv2_x(self.conv1(x)))))
        return self.fc(self.avg_pool(out).flatten(1))


class BasicBlockCIFAR(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNetCIFAR(nn.Module):
    """ResNet-20/32/... for CIFAR (three stages of 16/32/64 channels)."""

    def __init__(self, num_blocks, num_classes=10, width=16):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, num_blocks[0], 1)
        self.layer2 = self._make_layer(width * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(width * 4, num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 4, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlockCIFAR(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer3(self.layer2(self.layer1(out)))
        return self.fc(self.avgpool(out).flatten(1))


BUILTIN = {
    "resnet20": lambda num_classes: ResNetCIFAR([3, 3, 3], num_classes=num_classes),
    "resnet18": lambda num_classes: ResNet([2, 2, 2, 2], num_classes=num_classes),
    "resnet34": lambda num_classes: ResNet([3, 4, 6, 3], num_classes=num_classes),
}


# ----------------------------------------------------------------------------- registry

@dataclass
class ModelBundle:
    model: nn.Module
    data_config: DataConfig
    source: str
    name: str


class HFLogits(nn.Module):
    """``pixel_values -> logits`` so HF classifiers train and evaluate like any torch classifier."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


def _split(spec: str):
    if ":" not in spec:
        return "builtin", spec
    src, name = spec.split(":", 1)
    return src.lower(), name


def _replace_head(model: nn.Module, num_classes: int) -> None:
    """Swap the final classifier Linear of a torchvision model for ``num_classes`` outputs."""
    for attr in ("fc", "head", "heads", "classifier"):
        parent = getattr(model, attr, None)
        if parent is None:
            continue
        if isinstance(parent, nn.Linear):
            if parent.out_features != num_classes:
                setattr(model, attr, nn.Linear(parent.in_features, num_classes, bias=parent.bias is not None))
            return
        last_name, last = None, None
        for n, m in parent.named_modules():
            if isinstance(m, nn.Linear):
                last_name, last = n, m
        if last is not None:
            if last.out_features != num_classes:
                new = nn.Linear(last.in_features, num_classes, bias=last.bias is not None)
                *path, leaf = last_name.split(".")
                p = parent
                for q in path:
                    p = getattr(p, q)
                setattr(p, leaf, new)
            return
    raise ValueError(f"could not locate classifier head on {type(model).__name__}")


def _torchvision(name: str, num_classes: int, pretrained: bool, kw: Dict[str, Any]):
    import torchvision.models as tvm
    cifar_stem = bool(kw.pop("cifar_stem", False))
    dc = DataConfig()
    if pretrained:
        weights = tvm.get_model_weights(name).DEFAULT
        model = tvm.get_model(name, weights=weights, **kw)
        _replace_head(model, num_classes)
        t = weights.transforms()
        crop, resize = int(t.crop_size[0]), int(t.resize_size[0])
        dc = DataConfig(input_size=crop, mean=tuple(t.mean), std=tuple(t.std),
                        interpolation=t.interpolation.value, crop_pct=crop / resize)
    else:
        model = tvm.get_model(name, weights=None, num_classes=num_classes, **kw)
    if cifar_stem:
        if not hasattr(model, "conv1") or not hasattr(model, "maxpool"):
            raise ValueError("cifar_stem only applies to torchvision ResNets")
        c = model.conv1
        model.conv1 = nn.Conv2d(c.in_channels, c.out_channels, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
    return model, dc


def _timm(name: str, num_classes: int, pretrained: bool, kw: Dict[str, Any]):
    import timm
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kw)
    img = kw.get("img_size")
    if pretrained:
        cfg = timm.data.resolve_model_data_config(model)
        dc = DataConfig(input_size=int(img or cfg["input_size"][-1]), mean=tuple(cfg["mean"]), std=tuple(cfg["std"]),
                        interpolation=cfg.get("interpolation", "bicubic"), crop_pct=float(cfg.get("crop_pct", 0.875)))
    else:
        dc = DataConfig(input_size=int(img) if img else None)
    return model, dc


def _hf_size(v) -> Optional[int]:
    """HF processor ``size`` / ``crop_size``: int, dict or ``SizeDict`` (shortest_edge | height)."""
    if v is None or isinstance(v, int):
        return v
    for k in ("shortest_edge", "height"):
        x = v.get(k) if hasattr(v, "get") else getattr(v, k, None)
        if x:
            return int(x)
    return None


def _hf(name: str, num_classes: int, pretrained: bool, kw: Dict[str, Any]):
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    cfg = AutoConfig.from_pretrained(name, num_labels=num_classes, **kw)
    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(name, config=cfg, ignore_mismatched_sizes=True)
        proc = AutoImageProcessor.from_pretrained(name)
        size = _hf_size(proc.size)
        crop_size = _hf_size(getattr(proc, "crop_size", None)) or size
        dc = DataConfig(input_size=int(crop_size), mean=tuple(proc.image_mean), std=tuple(proc.image_std),
                        interpolation="bicubic" if int(getattr(proc, "resample", 2) or 2) == 3 else "bilinear",
                        crop_pct=float(crop_size) / float(size) if getattr(proc, "do_center_crop", False) else 1.0)
    else:
        model = AutoModelForImageClassification.from_config(cfg)
        dc = DataConfig(input_size=int(getattr(cfg, "image_size", 0)) or None)
    return HFLogits(model), dc


def build_model(
    spec: str,
    *,
    num_classes: int,
    pretrained: bool = False,
    img_size: Optional[int] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> ModelBundle:
    src, name = _split(spec)
    kw = dict(model_kwargs or {})
    if src == "builtin":
        if pretrained:
            raise ValueError("builtin models have no pretrained weights; use init_ckpt")
        if name not in BUILTIN:
            raise ValueError(f"builtin model {name!r} not in {sorted(BUILTIN)}")
        model, dc = BUILTIN[name](num_classes), DataConfig()
    elif src == "torchvision":
        model, dc = _torchvision(name, num_classes, pretrained, kw)
    elif src == "timm":
        model, dc = _timm(name, num_classes, pretrained, kw)
    elif src == "hf":
        model, dc = _hf(name, num_classes, pretrained, kw)
    else:
        raise ValueError(f"unknown model source {src!r} (builtin | torchvision | timm | hf)")
    if img_size:
        dc = DataConfig(input_size=int(img_size), mean=dc.mean, std=dc.std,
                        interpolation=dc.interpolation, crop_pct=dc.crop_pct)
    return ModelBundle(model=model, data_config=dc, source=src, name=name)
