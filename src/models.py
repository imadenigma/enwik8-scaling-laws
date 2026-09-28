import numpy as np
import torch
from torch import nn
import math

from flops import EmbeddingOnlyConfig, MLPConfig, RecurrentConfig


class Bigram(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.n_drop = 0
        self.emb = nn.Embedding(vocab, vocab)

    def forward(self, x):
        return self.emb(x)


class MLP(nn.Module):
    def __init__(self, vocab, n_ctx = 8, n_emb = 32, n_hidden = 256):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_drop = self.n_ctx - 1
        self.emb = nn.Embedding(vocab, n_emb)
        self.net = nn.Sequential(
            nn.Linear(n_ctx * n_emb, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, vocab)
        )
    def forward(self, x):
        idx = x.unfold(1, self.n_ctx, 1)
        e = self.emb(idx)
        return self.net(e.flatten(2))
class RNNCell(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.x2h = nn.Linear(d_in, d_hidden)
        self.h2h = nn.Linear(d_hidden, d_hidden)
    def forward(self, x, h):
        h = torch.tanh(self.x2h(x) + self.h2h(h))
        return h, h

class LSTMCell(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_in = d_in
        self.x2h = nn.Linear(d_in, 4 * d_hidden, bias=False)
        self.h2h = nn.Linear(d_hidden, d_hidden * 4)
    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.x2h(x) + self.h2h(h)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, (h,c)
class Recurrent(nn.Module):
    def __init__(self, vocab, cell_cls, d_emb=32, d_hidden=256):
        super().__init__()
        self.n_drop = 0
        self.d_hidden = d_hidden
        self.emb = nn.Embedding(vocab, d_emb)
        self.cell = cell_cls(d_emb, d_hidden)
        self.head = nn.Linear(d_hidden, vocab)

    def init_state(self, B, device):
        z = torch.zeros(B, self.d_hidden, device=device)
        return (z, z.clone()) if isinstance(self.cell, LSTMCell) else z

    def forward(self, x, state=None, return_state=False):
        e = self.emb(x)
        state = state if state is not None else self.init_state(x.shape[0], x.device)
        outs = []
        for t in range(x.shape[1]):
            h, state = self.cell(e[:, t], state)
            outs.append(h)
        logits = self.head(torch.stack(outs, dim=1))
        return (logits, state) if return_state else logits

class GRUCell(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.x2h = nn.Linear(d_in, 3 * d_hidden, False)
        self.h2h = nn.Linear(d_hidden, 3 * d_hidden)

    def forward(self, x, h):
        xr, xz, xn = self.x2h(x).chunk(3, dim=-1)
        hr, hz, hn = self.h2h(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        h = (1 - z) * n + z * h
        return h, h



BUILDERS = {
    EmbeddingOnlyConfig: lambda c: Bigram(c.vocab),
    MLPConfig: lambda c: MLP(c.vocab, c.n_ctx, c.d_emb, c.d_hidden),
    RecurrentConfig: lambda c: Recurrent(c.vocab, CELLS[c.cell], c.d_emb, c.d_hidden),
}

CELLS = {
    "rnn": RNNCell,
    "gru": GRUCell,
    "lstm": LSTMCell
}

def build(cfg):
    return BUILDERS[type(cfg)](cfg)