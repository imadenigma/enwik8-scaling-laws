import numpy as np
import torch
from torch import nn
import math

from flops import EmbeddingOnlyConfig, MLPConfig, RecurrentConfig
from flops import TransformerConfig


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

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k , v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        cos, sin = rope_cache(T, self.d_head, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        y = att @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, d))

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class Block(nn.Module):
    def __init__(self, d_model, n_head, mlp_mult = 4):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult, bias=False),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab, n_layer = 4, d_model = 128, n_head = 4, mlp_mult = 4):
        super().__init__()
        self.n_drop = 0
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, mlp_mult) for _ in range(n_layer)
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

def rope_cache(T, d_head, device, base = 10000.0):
    k = torch.arange(0, d_head, 2, device=device).float()
    theta = base ** (-k / d_head)
    pos = torch.arange(T, device=device).float()
    angle = pos[:, None] * theta[None, :]
    return angle.cos(), angle.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2] #even and odd components
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)

BUILDERS = {
    EmbeddingOnlyConfig: lambda c: Bigram(c.vocab),
    MLPConfig: lambda c: MLP(c.vocab, c.n_ctx, c.d_emb, c.d_hidden),
    RecurrentConfig: lambda c: Recurrent(c.vocab, CELLS[c.cell], c.d_emb, c.d_hidden),
    TransformerConfig: lambda c: Transformer(c.vocab, c.n_layer, c.d_model, c.n_head, c.mlp_mult)
}

CELLS = {
    "rnn": RNNCell,
    "gru": GRUCell,
    "lstm": LSTMCell
}

def build(cfg):
    return BUILDERS[type(cfg)](cfg)