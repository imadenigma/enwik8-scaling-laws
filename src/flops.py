from dataclasses import dataclass
import torch
from torch.utils.flop_counter import FlopCounterMode
import sys
import pathlib

FLOPS_PER_MAC = 2
BACKWARD_MULTIPLIER = 2

def linear_flops(positions, d_in, d_out, training = True):
    forward = FLOPS_PER_MAC * positions * d_in * d_out
    return forward * (1 + BACKWARD_MULTIPLIER) if training else forward

@dataclass(frozen=True)
class MLPConfig:
    vocab: int = 205
    n_ctx: int = 8
    d_emb: int = 32
    d_hidden: int = 256
    batch: int = 64
    block: int = 128
    name: str = "mlp"
    #@property means this is computed on access rather than stored
    @property
    def d_in(self):
        return self.n_ctx * self.d_emb

    @property
    def positions_per_step(self):
        return self.batch * (self.block - self.n_ctx + 1)

    @property
    def embedding_param(self):
        return self.d_emb * self.vocab

    @property
    def weight_param(self):
        return self.d_in * self.d_hidden + self.d_hidden * self.vocab

    @property
    def bias_param(self):
        return self.vocab + self.d_hidden

    @property
    def total_params(self):
        return self.embedding_param + self.weight_param + self.bias_param

@dataclass(frozen=True)
class EmbeddingOnlyConfig:
    vocab: int = 205
    batch: int = 64
    block: int = 128
    name: str = "bigram"

    @property
    def positions_per_step(self) -> int:
        return self.batch * self.block

    @property
    def weight_params(self) -> int:
        return 0

    @property
    def total_params(self) -> int:
        return self.vocab * self.vocab

@dataclass(frozen=True)
class RecurrentConfig:
    vocab: int = 205
    cell: str = "lstm"
    d_emb: int = 32
    d_hidden: int = 256
    batch: int = 64
    block: int = 128
    name: str = "lstm"

    @property
    def gates(self):
        return {
            "rnn": 1,
            "gru": 3,
            "lstm": 4
        }[self.cell]

    @property
    def positions_per_step(self):
        return self.batch * self.block

    @property
    def embedding_params(self):
        return self.d_emb * self.vocab


    @property
    def weight_params(self):
        g = self.gates
        return self.d_emb * self.d_hidden * g + self.d_hidden * g * self.d_hidden + self.d_hidden * self.vocab

    @property
    def bias_params(self):
        return self.gates *  self.d_hidden + self.vocab

    @property
    def total_params(self):
        return self.weight_params + self.bias_params + self.embedding_params

def mlp_flops(config: MLPConfig, training = True):
    p = config.positions_per_step
    parts = {
        "embedding": 0,
        "hidden" : linear_flops(p, config.d_in, config.d_hidden, training),
        "output": linear_flops(p, config.d_hidden, config.vocab, training)
    }
    parts["total"] = sum(parts.values())
    return parts

def embedding_only_flops(config: EmbeddingOnlyConfig, training = True):
    return {"embedding": 0, "total": 0}

FLOP_FNS = {MLPConfig: mlp_flops, EmbeddingOnlyConfig: embedding_only_flops}

def flops_per_step(config, training = True):
    return FLOP_FNS[type(config)](config, training)["total"]

def total_flops(config, steps, training = True):
    return flops_per_step(config, training) * steps

def steps_for_budget(config, budget: float):
    per_step = flops_per_step(config)
    if per_step == 0:
        raise ValueError(f"{config.name} consumes no matmul FLOPs; budget is undefined")
    return int(budget // per_step)

def check_6nd(cfg) -> None:
    exact = flops_per_step(cfg, training=True)
    rule = 6 * cfg.weight_param * cfg.positions_per_step
    assert exact == rule, f"{cfg.name}: layerwise {exact:,} != 6ND {rule:,}"

def measure_with_torch(cfg: MLPConfig) -> int:

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from train_mlp import MLP

    model = MLP(cfg.vocab, cfg.n_ctx, cfg.d_emb, cfg.d_hidden)
    x = torch.randint(0, cfg.vocab, (cfg.batch, cfg.block))
    y = torch.randint(0, cfg.vocab, (cfg.batch, cfg.block - cfg.n_ctx + 1))

    counter = FlopCounterMode(display=False)
    with counter:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab), y.reshape(-1)
        )
        loss.backward()
    return counter.get_total_flops()

def recurrent_flops(config: RecurrentConfig, training=True):
    p, g = config.positions_per_step, config.gates
    parts = {
        "embedding": 0,
        "x2h" : linear_flops(p, config.d_emb,g * config.d_hidden, training),
        "h2h": linear_flops(p, config.d_hidden, g * config.d_hidden, training),
        "head": linear_flops(p, config.d_hidden, config.vocab, training),
    }
    parts["total"] = sum(parts.values())
    return parts

FLOP_FNS[RecurrentConfig] = recurrent_flops

if __name__ == "__main__":
    cfg = MLPConfig()
    check_6nd(cfg)
    parts = mlp_flops(cfg)

    print(
        f"config: {cfg.name}  n_ctx={cfg.n_ctx} d_emb={cfg.d_emb} "
        f"d_hidden={cfg.d_hidden} batch={cfg.batch} block={cfg.block}"
    )
    print(f"positions/step   : {cfg.positions_per_step:,}")
    print(f"params total     : {cfg.total_params:,}")
    print(f"  embedding      : {cfg.embedding_param:,}  (excluded from N)")
    print(f"  weights (= N)  : {cfg.weight_param:,}")
    print(f"  biases         : {cfg.bias_param:,}  (excluded from N)")
    print()
    for k in ("embedding", "hidden", "output", "total"):
        share = parts[k] / parts["total"] * 100
        print(f"FLOPs {k:<10}: {parts[k]:>15,}  ({share:5.1f}%)")
    print()
    print(
        f"6ND check        : 6 x {cfg.weight_param:,} x "
        f"{cfg.positions_per_step:,} = {6 * cfg.weight_param * cfg.positions_per_step:,}"
    )
    print(f"5000 steps       : {total_flops(cfg, 5000):.3e} FLOPs")
    print(measure_with_torch(cfg))