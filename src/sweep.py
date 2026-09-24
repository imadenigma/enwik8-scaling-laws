import json
import math
from dataclasses import asdict

from flops import MLPConfig
from train import RESULTS, load_results, run, TrainConfig, append_result


def done_keys(path= RESULTS):
    return {
        (
            r["arch_type"],
            json.dumps(r["arch"], sort_keys=True),
            json.dumps(r["train"], sort_keys=True)
        )
        for r in load_results(path)
    }

def key_of(arch, tcfg):
    return (type(arch).__name__, json.dumps(asdict(arch), sort_keys=True), json.dumps(asdict(tcfg), sort_keys=True))

def safe_run(arch, tcfg):
    try:
        result = run(arch, tcfg, verbose=False)
    except (ValueError, RuntimeError)  as e:
        return {
            "arch_type" : type(arch).__name__,
            "arch" : asdict(arch),
            "train" : asdict(tcfg),
            "val_bpc": None,
            "failed" : str(e)
        }
    if not math.isfinite(result["val_bpc"]):
        result["val_bpc"], result["failed"] = None, "non-finite loss"
    return result


LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SWEEP_STEPS = 1000

def lr_sweep(arch, lrs=LRS, steps=SWEEP_STEPS, seed=1337):
    done = done_keys()
    for lr in lrs:
        tfcg = TrainConfig(lr, steps, seed=seed)
        if key_of(arch, tfcg) in done:
            print(f"skip lr = {lr:<8}")
            continue
        r = safe_run(arch, tfcg)
        append_result(r)
        print(f"lr={lr:<8} val bpc {r['val_bpc'] or 'diverged'}")

def check_interior(results):
    ok = [(r["train"]["lr"], r["val_bpc"]) for r in results if r["val_bpc"]]
    best_lr = min(ok, key=lambda x: x[1])[0]
    lrs = sorted(lr for lr, _ in ok)
    if best_lr in (lrs[0], lrs[-1]):
        print(f"WARNING: best lr {best_lr} is at the grid edge — extend the grid")
    return best_lr

def check_ranking_stability(arch, best_lr, lrs=LRS, short=SWEEP_STEPS, long=None):
    long = long or short * 5
    pair = sorted(lrs, key=lambda x: abs(math.log10(x) - math.log10(best_lr)))[:2]
    print(f"\nstability at {long} steps: {pair[0]:g} vs {pair[1]:g}")

    out = {}
    for lr in pair:
        r = safe_run(arch, TrainConfig(lr=lr, steps=long, seed=1337))
        append_result(r)
        out[lr] = r["val_bpc"]
        print(f"  lr={lr:<8g} val bpc {out[lr]}")

    if all(v is not None for v in out.values()):
        held = out[pair[0]] <= out[pair[1]]
        print(f"  short-run winner {'held' if held else 'DID NOT HOLD'}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=SWEEP_STEPS)
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--stability", action="store_true", default=True)
    a = p.parse_args()
    arch = MLPConfig(d_hidden=a.d_hidden)
    lr_sweep(arch, steps=a.steps)
    results = [
        r
        for r in load_results()
        if r["arch"] == asdict(arch)
        and r["train"]["steps"] == a.steps
    ]
    best = check_interior(results)
    print("best lr:", best)
    if a.stability and best is not None:
        check_ranking_stability(arch, best, short=a.steps)