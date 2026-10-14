"""
PCT trainer.

  hardware spec (truth tables, tile)   --compile_layout-->  PatternLayout (certified)
  model (builtin | torchvision | timm | hf)  --convert_model-->  pattern layers
  dataset (cifar10 | cifar100 | tiny_imagenet | imagenet1k), single GPU or DDP, AMP

``run(cfg)`` trains on one GPU, spawns DDP workers for ``gpus="0,1,2,3"``, or joins an
existing ``torchrun`` group.

Run directory ``<out_dir>/<dataset>/<model>/<run_name>``: config.json, convert_report.txt,
train.log, latest.pth / best.pth, results.json.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .convert import ConvertPolicy, convert_model, is_pattern_module
from .data import build_loaders, get_dataset_spec
from .layout import compile_layout
from .models import build_model
from .quant import ActQuantizer, quantize_symmetric_int, weight_scale
from .spec import HWSpec


@dataclass
class TrainConfig:
    # approximate hardware
    hw: Any = "configs/hw/or.yaml"         # hardware / compressor spec path (.yaml / .json) or inline dict
    w_bits: int = 8
    a_bits: int = 8
    no_projection: bool = False            # plain fake-quant STE QAT on the same layers (no pattern projection)
    # model / data
    model: str = "builtin:resnet20"
    dataset: str = "cifar10"
    data_root: str = ""
    pretrained: bool = False
    init_ckpt: str = ""                    # float or QAT checkpoint, loaded after conversion
    img_size: int = 0
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    keep_first: bool = True
    keep_last: bool = True
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    # optimization
    epochs: int = 10
    batch: int = 256                       # per GPU
    eval_batch: int = 0
    optimizer: str = "sgd"                 # sgd | adamw
    lr: float = 0.01
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    no_decay_1d: Optional[bool] = None     # exclude 1-d params from weight decay; default: True for adamw only
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    amp: str = "fp16"                      # fp16 | bf16 | none
    calib_batches: int = 10                # activation-range calibration when starting from trained weights
    # run
    seed: int = 42
    workers: int = 8
    gpus: str = "0"
    port: str = "12480"
    out_dir: str = "./runs"
    run_name: str = ""
    resume: bool = False
    eval_before: bool = True
    save_ckpt: bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**dict(d))


# ----------------------------------------------------------------------------- helpers

def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hw_label(hw: Any) -> str:
    if isinstance(hw, Mapping):
        return str(hw.get("name", "custom_hw"))
    return Path(hw).stem


def default_run_name(cfg: TrainConfig) -> str:
    init = "pt" if cfg.pretrained else ("ft" if cfg.init_ckpt else "scratch")
    hw = "noproj" if cfg.no_projection else hw_label(cfg.hw)
    return f"{hw}_w{cfg.w_bits}a{cfg.a_bits}_{init}_ep{cfg.epochs}_seed{cfg.seed}"


def run_dir(cfg: TrainConfig) -> Path:
    model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg.model)
    return Path(cfg.out_dir) / get_dataset_spec(cfg.dataset).name / model_tag / (cfg.run_name or default_run_name(cfg))


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items() if torch.is_tensor(v)}


def load_weights(model: nn.Module, path: str) -> int:
    """Load a float or QAT checkpoint into a converted model. Float checkpoints carry no
    activation-quantizer state, so only those keys may be missing. Returns their count."""
    sd = _extract_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "input_quantizer." not in k]
    if bad or unexpected:
        raise RuntimeError(f"init_ckpt {path} mismatch: missing={bad[:8]} unexpected={list(unexpected)[:8]}")
    return len(missing)


@torch.no_grad()
def count_projection_changes(model: nn.Module) -> int:
    """Quantized weight elements changed by the pattern projection."""
    total = 0
    for m in model.modules():
        if is_pattern_module(m) and m.layout is not None:
            q = quantize_symmetric_int(m.weight, weight_scale(m.weight, m.w_bits), m.w_bits)
            total += int((m.layout.project(q) != q).sum())
    return total


@torch.no_grad()
def calibrate_activations(model: nn.Module, loader, device, n_batches: int) -> None:
    """Initialize activation ranges on ``n_batches`` training batches (BN / dropout in eval mode)."""
    if n_batches <= 0:
        return
    was_training = model.training
    model.eval()
    for q in model.modules():
        if isinstance(q, ActQuantizer):
            q.initialized.fill_(False)
            q.train()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(device, non_blocking=True))
    model.train(was_training)


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    no_decay_1d = cfg.no_decay_1d if cfg.no_decay_1d is not None else cfg.optimizer == "adamw"
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if no_decay_1d and p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    raise ValueError(f"optimizer {cfg.optimizer!r} not in (sgd, adamw)")


def build_lr_scheduler(opt, epochs: int, warmup: int):
    S = torch.optim.lr_scheduler
    warmup = max(int(warmup), 0)
    if warmup <= 0:
        return S.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    if warmup >= epochs:
        return S.LinearLR(opt, start_factor=0.01, total_iters=epochs)
    return S.SequentialLR(opt, [S.LinearLR(opt, start_factor=0.01, total_iters=warmup),
                                S.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1))], milestones=[warmup])


def _all_reduce(vals: List[float], device, world: int) -> List[float]:
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    return t.tolist()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, world: int = 1, amp_dtype=None) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            out = model(x)
        correct += int((out.argmax(1) == y).sum())
        total += int(y.numel())
    c, t = _all_reduce([correct, total], device, world)
    return 100.0 * c / max(1.0, t)


class _Log:
    def __init__(self, path: Optional[Path], rank: int):
        self.rank, self.f = rank, (open(path, "a") if (path is not None and rank == 0) else None)

    def __call__(self, msg: str) -> None:
        if self.rank != 0:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


# ----------------------------------------------------------------------------- build

def build(cfg: TrainConfig, device: torch.device):
    """Compile the layout, build and convert the model. Returns (model, layout, report, data_config)."""
    layout = compile_layout(cfg.hw, bw=cfg.w_bits)
    ds = get_dataset_spec(cfg.dataset)
    bundle = build_model(cfg.model, num_classes=ds.num_classes, pretrained=cfg.pretrained,
                         img_size=cfg.img_size or None, model_kwargs=cfg.model_kwargs)
    model = bundle.model.to(device)
    report = convert_model(model, layout.to(device), w_bits=cfg.w_bits, a_bits=cfg.a_bits,
                           project=not cfg.no_projection,
                           policy=ConvertPolicy(include=cfg.include, exclude=cfg.exclude,
                                                keep_first=cfg.keep_first, keep_last=cfg.keep_last))
    if not report.converted:
        raise RuntimeError(f"no layers converted:\n{report.summary()}")
    return model, layout, report, bundle.data_config


# ----------------------------------------------------------------------------- train

def train(cfg: TrainConfig, rank: int = 0, world: int = 1, device: Optional[torch.device] = None) -> Dict[str, Any]:
    seed_all(cfg.seed + rank)
    if device is None:
        device = torch.device(f"cuda:{rank}" if world > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[cfg.amp]
    if device.type != "cuda":
        amp_dtype = None

    out = run_dir(cfg)
    if rank == 0 and cfg.save_ckpt:
        out.mkdir(parents=True, exist_ok=True)
    log = _Log(out / "train.log" if cfg.save_ckpt else None, rank)

    model, layout, report, data_config = build(cfg, device)
    loaders = build_loaders(cfg.dataset, batch=cfg.batch, workers=cfg.workers, root=cfg.data_root,
                            data_config=data_config, rank=rank, world=world,
                            eval_batch=cfg.eval_batch or None, seed=cfg.seed)
    if rank == 0:
        log(f"run_dir={out}")
        roles = {k: (s.free, s.tethered) for k, s in layout.segments.items()}
        log(f"hw={cfg.hw} tile={list(layout.tile)} width={layout.tile_width} bw={layout.bw} "
            f"(free, tethered)={roles} certified={layout.certificate.passed} projection={not cfg.no_projection}")
        log(f"data={asdict(loaders.data_config)}")
        log(report.summary())
        if cfg.save_ckpt:
            (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            (out / "convert_report.txt").write_text(report.summary() + "\n")

    if cfg.init_ckpt:
        n_missing = load_weights(model, cfg.init_ckpt)
        log(f"loaded init_ckpt={cfg.init_ckpt} (activation-quantizer keys missing: {n_missing})")

    hw_spec = HWSpec.load(cfg.hw).to_dict()
    opt = build_optimizer(cfg, model)
    lr_sched = build_lr_scheduler(opt, cfg.epochs, cfg.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start_epoch, best_acc, history, init_acc = 0, 0.0, [], None
    latest = out / "latest.pth"
    resumed = cfg.resume and latest.is_file()
    if resumed:
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        lr_sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch, best_acc = int(ck["epoch"]) + 1, float(ck["best_acc"])
        history, init_acc = ck.get("history", []), ck.get("init_acc")
        log(f"resumed from {latest} at epoch {start_epoch} (best={best_acc:.3f})")
    elif (cfg.pretrained or cfg.init_ckpt) and not all(
            bool(q.initialized) for q in model.modules() if isinstance(q, ActQuantizer)):
        calibrate_activations(model, loaders.train, device, cfg.calib_batches)

    net = nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model

    if cfg.eval_before and not resumed:
        init_acc = evaluate(net, loaders.val, device, world, amp_dtype)
        log(f"pre-train val acc={init_acc:.3f} projection_changes={count_projection_changes(model)}")

    for epoch in range(start_epoch, cfg.epochs):
        if loaders.train_sampler is not None:
            loaders.train_sampler.set_epoch(epoch)
        lr = opt.param_groups[0]["lr"]
        net.train()
        t0 = time.time()
        loss_sum = correct = seen = 0.0
        n_batches = len(loaders.train)
        for bi, (x, y) in enumerate(loaders.train):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = net(x)
                loss = criterion(logits, y)
            lv = float(loss.detach())
            if not math.isfinite(lv):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_sum += lv * y.numel()
            correct += float((logits.argmax(1) == y).sum())
            seen += y.numel()
            if rank == 0 and bi % 200 == 0:
                print(f"epoch {epoch + 1}/{cfg.epochs} batch {bi}/{n_batches} loss={lv:.4f}", flush=True)
        lr_sched.step()
        loss_sum, correct, seen = _all_reduce([loss_sum, correct, seen], device, world)
        val_acc = evaluate(net, loaders.val, device, world, amp_dtype)
        rec = dict(epoch=epoch, lr=lr, train_loss=loss_sum / max(1.0, seen),
                   train_acc=100.0 * correct / max(1.0, seen), val_acc=val_acc, time=time.time() - t0)
        history.append(rec)
        improved = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        log(f"epoch {epoch}/{cfg.epochs} loss={rec['train_loss']:.4f} train={rec['train_acc']:.2f} "
            f"val={val_acc:.3f} best={best_acc:.3f} lr={lr:.3g} t={rec['time']:.0f}s")
        if rank == 0 and cfg.save_ckpt:
            ck = dict(epoch=epoch, config=asdict(cfg), hw_spec=hw_spec,
                      model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                      scheduler_state_dict=lr_sched.state_dict(), scaler_state_dict=scaler.state_dict(),
                      best_acc=best_acc, val_acc=val_acc, init_acc=init_acc, history=history)
            torch.save(ck, latest)
            if improved:
                torch.save(ck, out / "best.pth")

    results = dict(best_acc=best_acc, final_acc=history[-1]["val_acc"] if history else init_acc, init_acc=init_acc,
                   epochs_run=len(history), projection_changes=count_projection_changes(model),
                   converted_layers=len(report.converted),
                   skipped_layers=[(r.name, r.reason) for r in report.skipped], run_dir=str(out))
    if rank == 0 and cfg.save_ckpt:
        (out / "results.json").write_text(json.dumps({**results, "history": history}, indent=2))
    log(f"DONE best_acc={best_acc:.3f} final={results['final_acc']}")
    return results


# ----------------------------------------------------------------------------- launch

def _worker(rank: int, world: int, cfg: TrainConfig) -> None:
    dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank)
    torch.cuda.set_device(rank)
    try:
        train(cfg, rank, world)
    finally:
        dist.destroy_process_group()


def run(cfg: TrainConfig) -> Optional[Dict[str, Any]]:
    if "LOCAL_RANK" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
        try:
            return train(cfg, rank, world, device=torch.device(f"cuda:{local}"))
        finally:
            dist.destroy_process_group()
    gpus = [g for g in cfg.gpus.split(",") if g.strip()]
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if len(gpus) <= 1:
        return train(cfg)
    import torch.multiprocessing as mp
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = cfg.port
    mp.spawn(_worker, args=(len(gpus), cfg), nprocs=len(gpus), join=True)
    res = run_dir(cfg) / "results.json"
    return json.loads(res.read_text()) if res.is_file() else None
