import numpy as np
import torch
from models import build
from data import get_batch
from train import loss_fn


def grad_norms_per_timestep(arch, steps = 200, seed = 1337):
    torch.manual_seed(seed)
    model = build(arch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    acc = torch.zeros(arch.block)
    for _ in range(steps):
        x, y =  get_batch("train", arch.batch, arch.block, rng)
        e = model.embedding(x)
        state, outs = model.init_state(x.shape[0], x.device), []
        for  t in range(x.shape[1]):
            h, state = model.cell(e[:,t], state)
            h.retain_grad()
            outs.append(h)
        loss = loss_fn(model, model.head(torch.stack(outs, dim=1)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        acc += torch.tensor([o.grad.norm().item() for o in outs])
        opt.step()
    return acc / steps

