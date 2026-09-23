import numpy as np
import torch
from torch import nn
import math

from flops import EmbeddingOnlyConfig, MLPConfig

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

BUILDERS = {
    EmbeddingOnlyConfig: lambda c: Bigram(c.vocab),
    MLPConfig: lambda c: MLP(c.vocab, c.n_ctx, c.d_emb, c.d_hidden),
}


def build(cfg):
    return BUILDERS[type(cfg)](cfg)