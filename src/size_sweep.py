import argparse
import pathlib

import numpy as np
import math

from flops import MLPConfig, flops_per_step
from train import RESULTS
from sweep import SWEEP_STEPS, LRS, runs_for, lr_sweep, check_interior

D_HIDDENS = [32, 64, 128, 256, 512, 1024]

def size_ladder(n_ctx = 8, d_emb = 32):
    return [MLPConfig(n_ctx=n_ctx, d_emb=d_emb, d_hidden=d, name=f"mlp_h{d}") for d in D_HIDDENS]

def preview_cost(archs, steps = SWEEP_STEPS, lrs = LRS):
    total = 0
    for a in archs:
        f = flops_per_step(a) * steps * len(lrs)
        total += f
        print(f"{a.name:<10} {a.total_params:>9,} params {a.weight_param:>9,} N {f:.2e} FLOPs")
    print(f"total: {total:.2e} FLOPs over {len(archs) * len(lrs)} runs")

def size_sweep(archs=None, steps = SWEEP_STEPS, path = RESULTS):
    archs = archs or size_ladder()
    best = {}
    for arch in archs:
        print(f"\n{arch.name}: {arch.total_params:,} params")
        lr_sweep(arch, steps=steps)
        best[arch.name] = check_interior(runs_for(arch, steps, path))
        print(f"best lr: {best[arch.name]}")

def report_lr_shift(best):
    print("width best lr")
    for name, lr in best.items():
        print(f"{name:<10} {lr if lr else "none" }")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=SWEEP_STEPS)
    p.add_argument("--preview", action="store_true",
                   help="print the compute cost and exit without training")
    p.add_argument("--max-hidden", type=int, default=None,
                   help="stop the ladder at this width")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    path = RESULTS if a.out is None else pathlib.Path(a.out)
    archs = size_ladder()
    if a.max_hidden:
        archs = [c for c in archs if c.d_hidden <= a.max_hidden]

    preview_cost(archs, steps=a.steps)
    if a.preview:
        raise SystemExit

    best = size_sweep(archs, steps=a.steps, path=path)
    report_lr_shift(best)