import numpy as np
import pytest
import torch
from torch.nn import functional as F

from noisynets.networks import NoisyLinear, QNetwork
from noisynets.replay import ReplayBuffer
from noisynets.train import Config, epsilon_at, evaluate, td_target


def test_noisy_layer_has_fixed_noise_until_resampled_and_mean_eval():
    torch.manual_seed(7)
    layer = NoisyLinear(3, 4)
    x = torch.ones(2, 3)
    layer.reset_noise()
    y = layer(x).detach()
    torch.testing.assert_close(y, layer(x))
    layer.reset_noise()
    assert not torch.allclose(y, layer(x))
    layer.eval()
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight_mu, layer.bias_mu))


def test_zero_sigma_is_linear_and_sigma_receives_gradient():
    layer = NoisyLinear(3, 2, sigma0=0)
    layer.reset_noise()
    x = torch.randn(5, 3)
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight_mu, layer.bias_mu))
    layer(x).square().mean().backward()
    assert layer.weight_sigma.grad.abs().sum() > 0


def test_factorisation_and_fixed_sigma_ablation():
    layer = NoisyLinear(5, 3, learn_sigma=False)
    layer.reset_noise()
    assert torch.linalg.matrix_rank(layer.weight_epsilon) == 1
    assert not layer.weight_sigma.requires_grad
    assert layer.weight_mu.requires_grad


def test_truncation_keeps_bootstrap_and_termination_stops_it():
    # Оба перехода завершают эпизод, но только первый является терминальным.
    target = td_target(torch.tensor([1., 1.]), torch.tensor([1., 0.]),
                       torch.tensor([10., 10.]), 0.9)
    torch.testing.assert_close(target, torch.tensor([1., 10.]))


def test_replay_wraparound_and_seeded_sampling():
    a, b = ReplayBuffer(3, 2, 9), ReplayBuffer(3, 2, 9)
    with pytest.raises(ValueError):
        a.sample(1)
    for i in range(5):
        for buffer in [a, b]:
            buffer.add([i, i], 0, i, [i+1, i+1], False)
    for x, y in zip(a.sample(3), b.sample(3)):
        torch.testing.assert_close(x, y)
    assert set(a.obs[:, 0]) == {2, 3, 4}


def test_identical_mean_parameter_initialisation():
    torch.manual_seed(10)
    baseline = QNetwork(4, 2)
    torch.manual_seed(10)
    noisy = QNetwork(4, 2, noisy=True)
    baseline.eval()
    noisy.eval()
    x = torch.randn(20, 4)
    torch.testing.assert_close(baseline(x), noisy(x))


def test_evaluation_preserves_model_buffers_and_rng():
    torch.manual_seed(13)
    model = QNetwork(4, 2, noisy=True)
    model.train()
    model.reset_noise()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    first = evaluate(model, "CartPole-v1", 2, 500, stochastic=True)
    second = evaluate(model, "CartPole-v1", 2, 500, stochastic=True)
    assert first == second and model.training
    torch.testing.assert_close(rng, torch.get_rng_state())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k])


def test_epsilon_schedule_and_invalid_configuration():
    c = Config()
    assert epsilon_at(0, c) == c.epsilon_start
    assert epsilon_at(999999, c) == pytest.approx(c.epsilon_end)
    with pytest.raises(ValueError):
        Config(train_frequency=0).validate()


def test_training_checkpoint_roundtrip(tmp_path):
    import json
    from noisynets.train import run
    from noisynets.evaluate import load_checkpoint
    output = tmp_path / "short-run"
    c = Config(algorithm="noisy", seed=17, steps=128, learning_starts=32,
               batch_size=16, replay_capacity=256, eval_interval=64,
               eval_episodes=2, final_episodes=2)
    run(c, output)
    assert (output / "COMPLETE").exists()
    model, config = load_checkpoint(output / "model.pt")
    saved = json.loads((output / "final_returns.json").read_text())
    assert evaluate(model, c.env, c.final_episodes, c.final_seed) == saved["mean_weights"]
    assert config["seed"] == c.seed
