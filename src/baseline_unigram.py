import pathlib

import numpy as np
from skimage.util import dtype

ROOT = pathlib.Path(__file__).resolve().parent.parent   # .../scaling-laws
DATA = ROOT / "data"

vocab = np.load(DATA / "vocab.npy")
V = len(vocab)
train = np.fromfile(DATA / "train.bin", dtype=np.uint8)
val = np.fromfile(DATA / "val.bin", dtype=np.uint8)

counts = np.bincount(train, minlength=V).astype(np.float64)
p = (counts + 1) / (counts.sum() + V)
print("uniform bpc:", np.log2(V))
print("unigram bpc:", -np.log2(p[val]).mean())