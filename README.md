# PCT: Pattern-Constrained Training for Approximate Compute-in-Memory

Digital compute-in-memory (DCIM) macros often reduce partial products with **approximate
compressors** (OR gates, approximate 4:2 / 3:2 compressors, ...) to save area and energy. These
compressors are wrong only for specific partial-product patterns. **PCT** turns a compressor's
truth table into a weight projection that makes those patterns unreachable *for every
activation*, certifies it, and fine-tunes the network under that projection. The approximate
hardware then computes exactly the same result as an exact adder tree.

You only provide the truth table (a JSON file). Everything else is derived automatically.

## How it works

For an `A`-input compressor fed by partial products `pp_i = w_i[b] AND x_i[b]` on bit plane `b`:

1. **Error predicate.** `E(pp) = [approx(pp) != popcount(pp)]`, read from the truth table.
2. **Minimal error patterns `M(E)`.** These are the submask-minimal input sets whose
   all-ones pattern triggers an error. Activations are uncontrolled, so an error is reachable
   on a plane iff the weight bits on that plane cover a pattern in `M(E)`.
3. **Free / tethered roles.** Any minimum hitting set of `M(E)` is a valid *tethered* set; the
   other inputs are *free*.
   - Free weights are never modified.
   - Each tethered weight is projected, in order, to the nearest value whose bits avoid every
     plane where the committed weights could complete an error pattern.
   - Each step is compiled into a lookup table indexed by the committed weights' bit
     signature, so projection is a few gathers per bit plane.
4. **Certificate.** Two checks run at compile time:
   - A bit-width-independent per-plane proof that no free bit vector leaves a reachable error.
   - An exhaustive value-level check at 4 bits for compressors with up to 4 inputs.
5. **Element-wise allocation.** Every tile (output channel, tile index, kernel tap) picks its
   own free set among all minimum hitting sets reachable by a symmetry of `M(E)`, keeping the
   largest-magnitude weights free. Roles are re-derived on every forward pass.
6. **Training.** Weights are quantized per output channel, projected, and dequantized. The
   straight-through estimator passes gradients to the float weights; activations are
   fake-quantized per tensor.

Weights are tiled along the reduction (input-channel) axis. A tile can mix several
compressors, e.g. `[SXAFA, SXAFA, OR]` covers 3 + 3 + 2 = 8 inputs. Input channels beyond the
last full tile, and the first and last layers of the network, stay exact.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.9 and PyTorch >= 2.1. `timm` and `transformers` are only needed for
`timm:` / `hf:` models.

## Describing the hardware

**Compressor spec** (`specs/*.json`). This is the full truth table, keyed by the partial
products `pp0 pp1 ... pp_{A-1}`, with the integer the compressor outputs as the value:

```json
{
  "name": "or",
  "arity": 2,
  "approx_table": {"00": 0, "01": 1, "10": 1, "11": 1}
}
```

The spec also accepts two optional fields:
- `"bit_order": "pp0_last"` for tables written as `x_A ... x_1`.
- `"output_weights": [2, 2, 1]` for string values listing several output bits
  (e.g. `"101"` for Cout/C/S).

**Hardware spec** (`configs/hw/*.yaml`). This tiles one or more compressors along the
reduction axis:

```yaml
name: lsac_sxafa
compressors:
  afa: specs/lsac_sxafa.json     # path, or an inline {arity, approx_table}
  or: specs/or.json
tile: [afa, afa, or]
```

A compressor JSON can also be passed directly wherever a hardware spec is expected.

Included specs:

| spec | inputs | tile width | free / tethered |
|---|---|---|---|
| `or` | 2 | 2 | 1 / 1 |
| `ha` | 4 | 4 | 3 / 1 |
| `venka` | 4 | 4 | 2 / 2 |
| `4to2` | 4 | 4 | 2 / 2 |
| `lsac_sxafa` (`[afa, afa, or]`) | 3 + 3 + 2 | 8 | 2 / 1 per AFA, 1 / 1 per OR |

To inspect what a spec compiles to (roles, certificate, converted layers):

```bash
python -m pct.train --hw specs/venka.json --model builtin:resnet20 --w_bits 4 --dry_run
```

## Training

Every field of `TrainConfig` (`pct/trainer.py`) is a command-line flag. `--config` loads a
YAML file of defaults, and flags override it. Runs are written to
`runs/<dataset>/<model>/<run_name>/`, which contains `config.json`, `train.log`,
`latest.pth` / `best.pth`, and `results.json`.

**CIFAR-10 / CIFAR-100 / Tiny ImageNet.** First train a W4A8 QAT baseline, then fine-tune it
with PCT:

```bash
python -m pct.train --config configs/exp/cifar10_resnet20_qat.yaml
python -m pct.train --config configs/exp/cifar10_resnet20_pct.yaml --hw configs/hw/venka.yaml
```

**ImageNet-1k.** Retrain from pretrained weights (W8A8):

```bash
python -m pct.train --config configs/exp/imagenet_resnet18_pct.yaml --hw configs/hw/lsac_sxafa.yaml
python -m pct.train --config configs/exp/imagenet_resnet50_pct.yaml --hw configs/hw/4to2.yaml
python -m pct.train --config configs/exp/imagenet_vit_b16_pct.yaml --hw configs/hw/ha.yaml
```

Useful flags:
- `--gpus 0,1,2,3` launches DDP; `batch` is per GPU. Under `torchrun`, the existing process
  group is used instead.
- `--data_root` points to the dataset. Defaults are under `./data`, and CIFAR is downloaded
  automatically.
- `--no_projection true` trains the same layers with plain fake-quant STE, as a reference.
- `--model` accepts `builtin:resnet20|resnet18|resnet34`, `torchvision:<name>`,
  `timm:<name>`, or `hf:<repo>`.
- `--include` / `--exclude` take regexes on layer names. `--keep_first` / `--keep_last`
  control which layers stay in floating point.

## Python API

```python
import torch
import torchvision
from pct import compile_layout, convert_model

layout = compile_layout("configs/hw/lsac_sxafa.yaml", bw=8)   # compiled + certified
print(layout.certificate.passed)                               # True

model = torchvision.models.resnet18(weights="DEFAULT")
report = convert_model(model, layout, w_bits=8, a_bits=8)      # in place
print(report.summary())

q = torch.randint(-128, 128, (64, 64, 3, 3), dtype=torch.int32)
q_legal = layout.project(q)                                    # error-free integer weights
```

## Repository layout

```
pct/
  spec.py          truth-table and hardware-spec parsing
  compiler.py      error patterns, free/tethered roles, step tables, certificate
  projection.py    closed-form nearest projection onto a forbidden-bit mask
  allocation.py    element-wise free/tethered allocation
  layout.py        tiled projection of a full weight tensor (PatternLayout)
  quant.py         weight / activation fake quantization, projected-weight STE
  nn.py            pattern-constrained Conv2d / Linear / MultiheadAttention
  convert.py       in-place model conversion with a per-layer report
  models.py        built-in CIFAR ResNets; torchvision / timm / HF loaders
  data.py          CIFAR-10/100, Tiny ImageNet, ImageNet-1k loaders
  trainer.py       training loop (single GPU / DDP, AMP, resume)
  train.py         command-line entry point (python -m pct.train)
specs/             compressor truth tables
configs/hw/        hardware specs (compressor tiles)
configs/exp/       experiment configs
```

## License

MIT, see [LICENSE](LICENSE).
