import numpy as np, pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent   # .../scaling-laws
DATA = ROOT / "data"

data = pathlib.Path(DATA / "enwik8").read_bytes()
assert len(data) == 100_000_000

arr = np.frombuffer(data, dtype=np.uint8)
vocab = np.unique(arr)
print("vocab size: ", len(vocab))

lookup = np.zeros(256, dtype=np.uint8)
lookup[vocab] = np.arange(len(vocab), dtype=np.uint8)
ids = lookup[arr]

splits = {
    "train": ids[ : 90_000_000],
    "val": ids[90_000_000 : 95_000_000],
    "test": ids[95_000_000 :]
}

for name, s in splits.items():
    s.tofile(DATA / f"{name}.bin")
    print(name, s.shape)
np.save(DATA / "vocab.npy", vocab)