import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

V = len(np.load(DATA / "vocab.npy"))
train = np.fromfile(DATA / "train.bin", dtype=np.uint8).astype(np.int64)
val = np.fromfile(DATA / "val.bin", dtype=np.uint8).astype(np.int64)

def contexts_and_targets(seq, order):
    n = len(seq) - order
    ctx = np.zeros(n, dtype=np.int64)
    for i in range(order):
        ctx = ctx * V + seq[i: i+n]
    return ctx, seq[order:]

def evaluate(order, ks=(1e-4,1e-3,1e-2,0.1,1.0)):
    tr_ctx, tr_tgt = contexts_and_targets(train, order)
    counts = np.bincount(tr_ctx * V + tr_tgt, minlength=V ** (order + 1)).reshape(V**order, V)
    rows = counts.sum(axis=1)

    va_ctx, va_tgt = contexts_and_targets(val, order)
    num_raw = counts[va_ctx, va_tgt].astype(np.float64)
    den_raw = rows[va_ctx].astype(np.float64)
    for k in ks:
        bpc = -np.log2((num_raw + k) / (den_raw + k * V)).mean()
        print(f"order={order}  k={k:<7}  val bpc = {bpc:.4f}")

for order in range(1, 2):
    evaluate(order)
