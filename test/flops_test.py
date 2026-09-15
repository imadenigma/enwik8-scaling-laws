import pytest

from src.flops import (
    EmbeddingOnlyConfig,
    MLPConfig,
    check_6nd,
    flops_per_step,
    linear_flops,
    steps_for_budget,
)


# --------------------------------------------------------------------------- #
#  Tiny config, worked by hand
#
#    vocab=4, n_ctx=2, d_emb=3, d_hidden=5, batch=2, block=6
#
#  positions = batch * (block - n_ctx + 1) = 2 * 5           = 10
#  d_in      = n_ctx * d_emb = 2 * 3                         = 6
#
#  hidden layer  Linear(6, 5):  6 * 10 * 6 * 5               = 1800
#  output layer  Linear(5, 4):  6 * 10 * 5 * 4               = 1200
#                                                     total  = 3000
#
#  weight params = 6*5 + 5*4 = 50   ->   6 * 50 * 10         = 3000  ✓
# --------------------------------------------------------------------------- #

TINY = MLPConfig(vocab=4, n_ctx=2, d_emb=3, d_hidden=5, batch=2, block=6, name="tiny")


def test_tiny_positions():
    assert TINY.positions_per_step == 10


def test_tiny_param_split():
    assert TINY.embedding_param == 12  # 4 * 3
    assert TINY.weight_param == 50  # 6*5 + 5*4
    assert TINY.bias_param == 9  # 5 + 4
    assert TINY.total_params == 71


def test_tiny_flops_hand_computed():
    assert flops_per_step(TINY) == 3000


def test_tiny_forward_only_is_one_third():
    assert flops_per_step(TINY, training=False) == 1000


def test_single_linear():
    # 2 FLOPs per MAC, forward only
    assert linear_flops(10, 6, 5, training=False) == 2 * 10 * 6 * 5


def test_6nd_identity_holds_across_configs():
    for cfg in [
        TINY,
        MLPConfig(),
        MLPConfig(n_ctx=2),
        MLPConfig(n_ctx=16, d_emb=64, d_hidden=512),
        MLPConfig(d_hidden=1024, batch=32, block=256),
    ]:
        check_6nd(cfg)


def test_default_config_matches_pytorch_param_count():
    """sum(p.numel() for p in MLP(205).parameters()) reported 125,037."""
    assert MLPConfig().total_params == 125_037


def test_embedding_only_consumes_no_matmul_flops():
    assert flops_per_step(EmbeddingOnlyConfig()) == 0


def test_budget_undefined_for_zero_flop_model():
    with pytest.raises(ValueError):
        steps_for_budget(EmbeddingOnlyConfig(), 1e12)


def test_steps_for_budget_round_trips():
    cfg = MLPConfig()
    budget = flops_per_step(cfg) * 137
    assert steps_for_budget(cfg, budget) == 137


def test_wider_hidden_costs_more_per_position():
    narrow = MLPConfig(d_hidden=128)
    wide = MLPConfig(d_hidden=512)
    assert flops_per_step(wide) > flops_per_step(narrow)


def test_longer_context_costs_more_but_yields_fewer_positions():
    short = MLPConfig(n_ctx=4)
    long = MLPConfig(n_ctx=64)
    assert long.d_in > short.d_in
    assert long.positions_per_step < short.positions_per_step
