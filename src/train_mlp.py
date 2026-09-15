import torch
import torch.nn as nn
import numpy as np
import math
from data import DATA, get_batch

N_CTX, D_EMB, D_HIDDEN = 8, 32, 256
LR, STEPS = 1e-3, 5000
EVAL_EVERY, EVAL_BATCHES = 250, 50
BATCH, BLOCK = 64, 128

LN2 = math.log(2)
V = len(np.load(DATA / "vocab.npy"))

class MLP(nn.Module):
    def __init__(self, V, n_ctx = 8, d_emb = 32, d_hidden = 256):
        super().__init__()
        self.n_ctx = n_ctx
        self.embedding = nn.Embedding(V, d_emb)
        self.net = nn.Sequential(
            nn.Linear(d_emb * n_ctx, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, V)
        )

    def forward(self, x):
        idx = x.unfold(1, self.n_ctx, 1)
        e = self.embedding(idx)
        return self.net(e.flatten(2))

model = MLP(V, N_CTX, D_EMB, D_HIDDEN)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def loss_fn(logits, y):
    y = y[:, model.n_ctx - 1 :]
    return nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

@torch.no_grad()
def evaluate(split, batches=EVAL_BATCHES):
    rng = np.random.default_rng(0)
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(split, BATCH, BLOCK, rng)
        total += loss_fn(model(x), y).item()
    model.train()
    return total / batches / LN2
if __name__ == "__main__":
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  (n_ctx={N_CTX} d_emb={D_EMB} d_hidden={D_HIDDEN})")
    print(f"before training, val bpc = {evaluate('val'):.4f}")
    rng = np.random.default_rng(1337)
    best = float('inf')
    for step in range(1, STEPS + 1):
        x,y = get_batch("train", BATCH, BLOCK, rng)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            val = evaluate("val")
            best = min(best, val)
            print(f"step {step}: loss = {loss:.4f}, best = {best:.4f}")
    print(f"\nbest val bpc: {best:.4f}  ({n_params:,} params)")
