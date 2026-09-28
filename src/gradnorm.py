import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import get_batch
from flops import RecurrentConfig
from models import build
from train import loss_fn

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _forward_retaining(model, x):
    e = model.emb(x)
    state = model.init_state(x.shape[0], x.device)
    outs = []
    for t in range(x.shape[1]):
        h, state = model.cell(e[:, t], state)
        h.retain_grad()
        outs.append(h)
    return model.head(torch.stack(outs, dim=1)), outs


def grad_norms_per_timestep(arch, steps=100, warmup=0, lr=1e-3, seed=1337):
    torch.manual_seed(seed)
    model = build(arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for _ in range(warmup):
        x, y = get_batch("train", arch.batch, arch.block, rng)
        loss = loss_fn(model, model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    acc = torch.zeros(arch.block)
    for _ in range(steps):
        x, y = get_batch("train", arch.batch, arch.block, rng)
        logits, outs = _forward_retaining(model, x)
        loss = loss_fn(model, logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        acc += torch.tensor([o.grad.norm().item() for o in outs])
        # no opt.step(): the measurement must not move the model it measures

    return (acc / steps).tolist()


def plot(norms, title, path):
    """x-axis is backprop depth, so gradient decay reads left to right."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, v in norms.items():
        depth = np.arange(len(v))[::-1]          # h_0 is the deepest
        ax.plot(depth, v, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("backprop depth (steps back from the loss)")
    ax.set_ylabel(r"$\|\partial L / \partial h_t\|$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def decay_ratio(v):
    return v[0] / v[-1]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    cells = ("rnn", "gru", "lstm")
    out = {}

    for label, warmup in (("init", 0), ("trained", a.warmup)):
        norms = {}
        for c in cells:
            arch = RecurrentConfig(cell=c, name=c, block=a.block)
            print(f"{label}: {c} (warmup={warmup})")
            norms[c] = grad_norms_per_timestep(
                arch, steps=a.steps, warmup=warmup, lr=a.lr)
            print(f"  depth-{a.block-1} / depth-0 ratio: {decay_ratio(norms[c]):.3f}")
        out[label] = norms
        plot(norms,
             f"Gradient norm by backprop depth ({label}"
             + (f", {warmup} steps" if warmup else "") + ")",
             RESULTS / f"gradnorm_{label}.png")

    json.dump(out, open(RESULTS / "gradnorm.json", "w"), indent=1)
    print("wrote", RESULTS / "gradnorm.json")

    print("\ndecay ratios (max depth / min depth)")
    for label in out:
        for c in cells:
            print(f"  {label:<8} {c:<5} {decay_ratio(out[label][c]):.3f}")