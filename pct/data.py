"""
Datasets and loaders: CIFAR-10/100, Tiny ImageNet, ImageNet-1k.

Transforms follow the model's data config (input size, mean/std, interpolation, crop_pct),
so pretrained timm / HF / torchvision models see their own preprocessing. Small images are
resized up when the model expects a larger input (e.g. ViT-224 on CIFAR).

Validation covers the full split: under DDP each rank evaluates a disjoint, unpadded shard.

Expected layouts under ``data_root`` (defaults in ``DATASETS``):
  cifar10 / cifar100   torchvision format (downloaded if missing)
  tiny_imagenet        tiny-imagenet-200/{train/<wnid>/..., val/images, val/val_annotations.txt}
  imagenet1k           {train,val}/<wnid>/*.JPEG
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    native_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    default_root: str


DATASETS = {
    "cifar10": DatasetSpec("cifar10", 10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), "data/cifar10"),
    "cifar100": DatasetSpec("cifar100", 100, 32, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), "data/cifar100"),
    "tiny_imagenet": DatasetSpec("tiny_imagenet", 200, 64, IMAGENET_MEAN, IMAGENET_STD, "data/tiny-imagenet-200"),
    "imagenet1k": DatasetSpec("imagenet1k", 1000, 224, IMAGENET_MEAN, IMAGENET_STD, "data/imagenet"),
}
ALIASES = {"imagenet": "imagenet1k", "imagenet-1k": "imagenet1k", "tiny-imagenet": "tiny_imagenet"}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[key]


@dataclass(frozen=True)
class DataConfig:
    """Model-side preprocessing. ``None`` fields fall back to the dataset defaults."""

    input_size: Optional[int] = None
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None
    interpolation: str = "bilinear"
    crop_pct: float = 0.875


_INTERP = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC,
           "nearest": T.InterpolationMode.NEAREST}


def resolve_data_config(ds: DatasetSpec, dc: Optional[DataConfig]) -> DataConfig:
    dc = dc or DataConfig()
    return replace(dc, input_size=int(dc.input_size or ds.native_size),
                   mean=tuple(dc.mean or ds.mean), std=tuple(dc.std or ds.std))


def build_transforms(ds: DatasetSpec, dc: DataConfig) -> Tuple[T.Compose, T.Compose]:
    size, interp = int(dc.input_size), _INTERP[dc.interpolation]
    norm = [T.ToTensor(), T.Normalize(dc.mean, dc.std)]
    if ds.name == "imagenet1k":
        train = [T.RandomResizedCrop(size, interpolation=interp), T.RandomHorizontalFlip()]
        val = [T.Resize(int(round(size / dc.crop_pct)), interpolation=interp), T.CenterCrop(size)]
    else:
        pad = 4 if ds.native_size == 32 else 8
        train = [T.RandomCrop(ds.native_size, padding=pad), T.RandomHorizontalFlip()]
        val = []
        if size != ds.native_size:
            train.append(T.Resize(size, interpolation=interp))
            val.append(T.Resize(size, interpolation=interp))
    return T.Compose(train + norm), T.Compose(val + norm)


class TinyImageNetVal(Dataset):
    """Tiny ImageNet validation split (flat image folder + annotation file)."""

    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.transform = transform
        images = os.path.join(root, "val", "images")
        self.samples = []
        with open(os.path.join(root, "val", "val_annotations.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2 or parts[1] not in class_to_idx:
                    continue
                path = os.path.join(images, parts[0])
                if os.path.isfile(path):
                    self.samples.append((path, class_to_idx[parts[1]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform is not None else img), y


def build_datasets(ds: DatasetSpec, root: str, tf_train, tf_val) -> Tuple[Dataset, Dataset]:
    root = os.path.expanduser(root or ds.default_root)
    if ds.name in ("cifar10", "cifar100"):
        cls = torchvision.datasets.CIFAR10 if ds.name == "cifar10" else torchvision.datasets.CIFAR100
        return (cls(root=root, train=True, download=True, transform=tf_train),
                cls(root=root, train=False, download=True, transform=tf_val))
    train = torchvision.datasets.ImageFolder(os.path.join(root, "train"), transform=tf_train)
    if ds.name == "tiny_imagenet":
        return train, TinyImageNetVal(root, train.class_to_idx, transform=tf_val)
    return train, torchvision.datasets.ImageFolder(os.path.join(root, "val"), transform=tf_val)


class ShardSampler(Sampler):
    """Disjoint, unpadded ``rank::world`` shard, so summed counts cover the split exactly once."""

    def __init__(self, n: int, rank: int, world: int):
        self.indices = list(range(rank, n, world))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    train_sampler: Optional[DistributedSampler]
    data_config: DataConfig


def build_loaders(
    dataset: str,
    *,
    batch: int,
    workers: int,
    root: str = "",
    data_config: Optional[DataConfig] = None,
    rank: int = 0,
    world: int = 1,
    eval_batch: Optional[int] = None,
    seed: int = 0,
) -> Loaders:
    """``batch`` is per process."""
    ds = get_dataset_spec(dataset)
    dc = resolve_data_config(ds, data_config)
    tf_train, tf_val = build_transforms(ds, dc)
    trainset, valset = build_datasets(ds, root, tf_train, tf_val)
    train_sampler = DistributedSampler(trainset, num_replicas=world, rank=rank, shuffle=True, seed=seed) \
        if world > 1 else None
    g = torch.Generator()
    g.manual_seed(seed)
    train = DataLoader(trainset, batch_size=batch, shuffle=train_sampler is None, sampler=train_sampler,
                       num_workers=workers, pin_memory=True, drop_last=True, generator=g,
                       persistent_workers=workers > 0)
    val = DataLoader(valset, batch_size=eval_batch or batch, sampler=ShardSampler(len(valset), rank, world),
                     num_workers=workers, pin_memory=True)
    return Loaders(train=train, val=val, train_sampler=train_sampler, data_config=dc)
