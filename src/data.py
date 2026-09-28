import numpy as np
import torch
import pathlib

DATA = pathlib.Path(__file__).parent.parent / "data"
_cache = {}

def get_split(split):
    if split not in _cache:
        _cache[split] = np.memmap(DATA / f"{split}.bin", dtype=np.uint8, mode="r")
    return _cache[split]

def get_batch(split, batch_size, block_size, rng, device="cpu"):
    data = get_split(split)
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i+1 + block_size] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if torch.device(device).type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
