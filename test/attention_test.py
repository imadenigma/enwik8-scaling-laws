import math

import torch

from src.models import CausalSelfAttention, rope_cache, apply_rope, Transformer


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

def test_rope_preserves_norm():
    x = torch.randn(1, 2, 8, 16)
    cos, sin = rope_cache(8, 16, x.device)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)

def test_rope_is_relative():
    """q_i . k_j depends only on i-j."""
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 8, 16), torch.randn(1, 1, 8, 16)
    cos, sin = rope_cache(8, 16, q.device)
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    a = (qr[0, 0, 3] * kr[0, 0, 1]).sum()
    qs, ks = apply_rope(q.roll(2, dims=2), cos, sin), apply_rope(k.roll(2, dims=2), cos, sin)
    b = (qs[0, 0, 5] * ks[0, 0, 3]).sum()
    assert torch.allclose(a, b, atol=1e-4)

def test_position_matters():
    torch.manual_seed(0)
    m = CausalSelfAttention(16, 4).eval()
    x = torch.randn(1, 4, 16)
    x2 = x.clone(); x2[0, 0] = x[0, 1]; x2[0, 1] = x[0, 0]   # swap first two
    with torch.no_grad():
        assert not torch.allclose(m(x)[0, 2], m(x2)[0, 2], atol=1e-5)

def test_transformer_shapes_and_causality():
    torch.manual_seed(0)
    m = Transformer(205, n_layer=2, d_model=32, n_head=4).eval()
    x = torch.randint(0, 205, (2, 10))
    with torch.no_grad():
        a = m(x)
        assert a.shape == (2, 10, 205)
        x2 = x.clone(); x2[0, 7] = (x[0, 7] + 1) % 205
        b = m(x2)
        assert not torch.allclose(a[0, :7], b[0, :7], atol=1e-6)
