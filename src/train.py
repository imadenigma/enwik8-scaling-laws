import json
import math
import pathlib
from dataclasses import asdict, dataclass
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from data import get_batch
from flops import EmbeddingOnlyConfig, MLPConfig, flops_per_step
from models import build

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "runs.jsonl"
LN2 = math.log(2)

@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    steps: int = 5000
    seed: int = 1337
    eval_every: int = 250
    eval_batches: int = 50

def loss_fn(model, logits, y):
    y = y[:, model.n_drop:]
    assert logits.shape[1] == y.shape[1], (logits.shape, y.shape)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())

@torch.no_grad()
def evaluate(model, arch, tfcg, split="val"):
    rng = np.random.default_rng(0)
    model.eval()
    total = 0.0
    for _ in range(tfcg.eval_batches):
        x, y = get_batch(split, arch.batch, arch.block, rng)
        total += loss_fn(model, model(x), y).item()
    model.train()
    return total / tfcg.eval_batches / LN2

def run(arch, tcfg=TrainConfig(), verbose=True):
    torch.manual_seed(tcfg.seed)
    model = build(arch)
    per_step = flops_per_step(arch)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    rng = np.random.default_rng(tcfg.seed)

    if verbose:
        print(f"{arch.name}: {arch.total_params:,} params, "
              f"{per_step:.3e} FLOPs/step, lr={tcfg.lr}, seed={tcfg.seed}")
        print(f"  step 0  val bpc {evaluate(model, arch, tcfg):.4f}")

    curve = []
    for step in range(1, tcfg.steps + 1):
        x, y = get_batch("train", arch.batch, arch.block, rng)
        loss = loss_fn(model, model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % tcfg.eval_every == 0:
            val = evaluate(model, arch, tcfg)
            curve.append((step, step * per_step, val))
            if verbose:
                print(f"  step {step:5d}  train bpc {loss.item()/LN2:.4f}  "
                      f"val bpc {val:.4f}")

    return {
        "arch_type": type(arch).__name__,
        "arch": asdict(arch),
        "train": asdict(tcfg),
        "val_bpc": min(c[2] for c in curve),
        "final_bpc": curve[-1][2],
        "flops": tcfg.steps * per_step,
        "weight_params": arch.weight_param,
        "total_params": arch.total_params,
        "curve": curve,
    }

def append_result(result, path=RESULTS):
    """Append-only. A crashed sweep leaves every completed run intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")
def load_results(path=RESULTS):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
if __name__ == "__main__":

    ARCHS = {
        "bigram": EmbeddingOnlyConfig(),
        "mlp": MLPConfig(),
    }

    p = argparse.ArgumentParser()
    p.add_argument("arch", nargs="?", default="mlp", choices=ARCHS)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    result = run(ARCHS[a.arch], TrainConfig(lr=a.lr, steps=a.steps, seed=a.seed))
    print(
        f"\nbest val bpc {result['val_bpc']:.4f}  "
        f"({result['flops']:.3e} FLOPs, {result['total_params']:,} params)"
    )

    if not a.no_save:
        append_result(result)
        print(f"appended to {RESULTS}")


