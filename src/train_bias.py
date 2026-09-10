import numpy as np
import torch
import torch.nn as nn
from data import get_batch, DATA
import math

torch.manual_seed(1337)
V = len(np.load(DATA / "vocab.npy"))

BATCH, BLOCK, STEPS = 64, 128, 2000
ln2 = math.log(2)

class BiasOnly(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(V))
    def forward(self, x):
        B, T = x.shape
        return self.bias.expand(B,T,-1)

model = BiasOnly(V)
opt = torch.optim.Adam(model.parameters(), lr=0.1)


def loss_fn(logits, y):
    return nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

@torch.no_grad()
def evaluate(split, batches = 50):
    rng = np.random.default_rng(0)
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = get_batch(split, BATCH, BLOCK, rng)
        tot += loss_fn(model(x), y).item()
    model.train()
    return tot / batches / ln2

print(f"before training, val bpc = {evaluate('val'):.4f}")

rng = np.random.default_rng(1337)
for step in range(1, STEPS + 1):
    x, y = get_batch("train", BATCH, BLOCK, rng)
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"step {step:5d}  train bpc {loss.item()/ln2:.4f}  val bpc {evaluate('val'):.4f}")

train = np.fromfile(DATA / "train.bin", dtype=np.uint8)
empirical = np.bincount(train, minlength=V) / len(train)
learned = torch.softmax(model.bias, dim=0).detach().numpy()
print("max abs difference:", np.abs(learned - empirical).max())