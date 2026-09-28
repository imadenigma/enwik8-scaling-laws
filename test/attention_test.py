import math

import torch

from src.models import CausalSelfAttention


def naive_attention(module, x):
    B, T, d = x.shape
    H, dh = module.n_head, module.d_head
    q, k, v = module.qkv(x).chunk(3, dim=-1)
    out = torch.zeros(B, T, d)
    for b in range(B):
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            for i in range(T):
                scores = [q[b, i, sl] @ k[b, j, sl] / math.sqrt(dh) for j in range(i + 1)]
                w = torch.softmax(torch.stack(scores), 0)
                out[b, i, sl] = sum(w[j] * v[b, j, sl] for j in range(i + 1))
    return module.proj(out)

def test_matches_naive():
    torch.manual_seed(0)
    m = CausalSelfAttention(16, 4).eval()
    with torch.no_grad():
        x1 = torch.randn(1, 1, 16)
        q, k, v = m.qkv(x1).chunk(3, dim=-1)
        print("T=1 fast:", torch.allclose(m(x1), m.proj(v), atol=1e-6))
        x = torch.randn(2, 6, 16)
        print("T=6 match:", torch.allclose(m(x), naive_attention(m, x), atol=1e-5))

def test_is_causal():
    """Changing input at t+1 must not change the output at t."""
    torch.manual_seed(0)
    m = CausalSelfAttention(16, 4).eval()
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        a = m(x)
        x2 = x.clone(); x2[0, 5] = torch.randn(16)
        b = m(x2)
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-6)
    assert not torch.allclose(a[0, 5], b[0, 5], atol=1e-6)