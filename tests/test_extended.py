import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from noisynets.networks import NoisyLinear, QNetwork
from noisynets.train import Config, evaluate_details
import noisynets.train as training


@pytest.mark.parametrize("cadence", ["action", "update"])
def test_behavior_noise_cadence_and_independent_target(monkeypatch, tmp_path, cadence):
    snapshots = {"behavior": [], "loss": [], "target": []}
    created = []

    class ObservedNetwork(QNetwork):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def forward(self, states):
            if self is created[0]:
                key = "behavior" if len(states) == 1 else "loss"
            else:
                key = "target"
            snapshots[key].append(self.layers[2].weight_epsilon.clone())
            return super().forward(states)

    class ShortEnvironment:
        observation_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)

        def reset(self, seed=None):
            self.length = 0
            return np.ones(6, dtype=np.float32), {}

        def step(self, action):
            self.length += 1
            return np.ones(6, dtype=np.float32), -1.0, False, self.length == 3, {}

        def close(self):
            pass

    monkeypatch.setattr(training, "QNetwork", ObservedNetwork)
    monkeypatch.setattr(training.gym, "make", lambda name: ShortEnvironment())
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: [-3.0])
    monkeypatch.setattr(training, "evaluate_details", lambda *args, **kwargs: [
        {"seed": 20000, "return": -3.0, "length": 3, "terminated": False, "truncated": True}
    ])
    c = Config(env="Acrobot-v1", algorithm="noisy", hidden=8, steps=12,
               learning_starts=4, batch_size=2, replay_capacity=20,
               train_frequency=4, target_interval=8, noise_cadence=cadence,
               eval_interval=6, eval_episodes=1, final_episodes=1)
    output = tmp_path / cadence
    training.run(c, output)

    behavior = snapshots["behavior"]
    assert len(behavior) == 12 and len(snapshots["loss"]) == len(snapshots["target"]) == 3
    # Разминка одинакова для обоих режимов: новый шум перед каждым действием.
    assert all(not torch.equal(a, b) for a, b in zip(behavior[:3], behavior[1:4]))
    if cadence == "update":
        # Между обновлениями сохраняется шум, в том числе при сбросе среды.
        for block in [behavior[4:8], behavior[8:12]]:
            assert all(torch.equal(block[0], value) for value in block[1:])
        assert not torch.equal(behavior[3], behavior[4])
        assert not torch.equal(behavior[7], behavior[8])
        assert not torch.equal(snapshots["loss"][0], behavior[4])
    else:
        assert all(not torch.equal(a, b) for a, b in zip(behavior, behavior[1:]))
    assert all(not torch.equal(a, b) for a, b in zip(snapshots["loss"], snapshots["target"]))

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["gradient_updates"] == 3
    for filename, recorded in metadata["training_source_sha256"].items():
        source = Path(training.__file__).with_name(filename)
        assert recorded == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("stochastic", [False, True])
def test_episode_details_preserve_model_and_rng(stochastic):
    torch.manual_seed(27)
    model = QNetwork(4, 2, noisy=True)
    model.train(not stochastic)
    model.reset_noise()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    first = evaluate_details(model, "CartPole-v1", 2, 51000, stochastic=stochastic)
    assert first == evaluate_details(model, "CartPole-v1", 2, 51000, stochastic=stochastic)
    assert model.training == (not stochastic)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [row["seed"] for row in first] == [51000, 51001]
    for row in first:
        assert row["length"] == row["return"] > 0
        assert row["terminated"] or row["truncated"]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())


def test_fixed_sigma_survives_optimizer_step():
    torch.manual_seed(3)
    layer = NoisyLinear(5, 3, learn_sigma=False)
    layer.reset_noise()
    before = {name: value.detach().clone() for name, value in layer.named_parameters()}
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(torch.randn(8, 5)).square().mean().backward()
    optimizer.step()
    assert torch.equal(layer.weight_sigma, before["weight_sigma"])
    assert torch.equal(layer.bias_sigma, before["bias_sigma"])
    assert not torch.equal(layer.weight_mu, before["weight_mu"])
    assert layer.weight_sigma.grad is None and layer.bias_sigma.grad is None


def test_acrobot_network_dimensions_and_parameter_counts():
    env = gym.make("Acrobot-v1")
    try:
        obs, _ = env.reset(seed=4)
        assert obs.shape == (6,) and env.action_space.n == 3
        for noisy, expected in [(False, 17795), (True, 34694)]:
            model = QNetwork(6, 3, hidden=128, noisy=noisy)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected
            assert model(torch.as_tensor(obs).unsqueeze(0)).shape == (1, 3)
        fixed = QNetwork(6, 3, hidden=128, noisy=True, learn_sigma=False)
        assert sum(p.numel() for p in fixed.parameters() if p.requires_grad) == 17795
    finally:
        env.close()


def test_invalid_noise_cadence_is_rejected():
    with pytest.raises(ValueError, match="noise_cadence"):
        Config(noise_cadence="episode").validate()


def test_cartpole_failure_penalty_is_only_used_in_training(tmp_path):
    config = Config(steps=75, hidden=8, batch_size=8, learning_starts=8,
                    target_tau=0.01, learning_rate_end_fraction=0.1,
                    sutton_barto_reward=True, eval_interval=75, eval_episodes=1,
                    final_episodes=2, epsilon_end=1.0)
    output = tmp_path / "cartpole"
    training.run(config, output)
    episodes = (output / "episodes.csv").read_text().splitlines()
    assert len(episodes) > 1
    assert all(line.split(",")[2] == "-1.0" for line in episodes[1:])
    final = json.loads((output / "final_episode_details.json").read_text())
    assert all(row["return"] == row["length"] > 0 for row in final["mean_weights"])
