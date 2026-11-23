import json

import pytest
import torch

import noisynets.train as training


@pytest.mark.parametrize("algorithm,patience,expected_step,expected_reason", [
    ("dqn", 2, 20, "validation_threshold"),
    ("noisy", 2, 40, "validation_threshold"),
    ("dqn", 0, 60, "budget_exhausted"),
])
def test_stopping_uses_validation_only_and_saves_actual_step(
        tmp_path, monkeypatch, algorithm, patience, expected_step, expected_reason):
    mean_calls = []
    stochastic_calls = []

    def evaluate(model, env_id, episodes, seed, stochastic=False):
        assert seed == 76050
        calls = stochastic_calls if stochastic else mean_calls
        index = len(calls)
        calls.append(seed)
        # У Noisy второй контроль сбрасывает счётчик, хотя оценка по μ успешна.
        score = 100.0 if stochastic and index == 2 else 500.0
        return [score] * episodes

    def final_details(model, env_id, episodes, seed, stochastic=False):
        assert seed == 86000
        assert len(mean_calls) == expected_step // 10 + 1
        return [{"seed": seed+i, "return": 1.0, "length": 1,
                 "terminated": True, "truncated": False} for i in range(episodes)]

    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr(training, "evaluate_details", final_details)
    c = training.Config(algorithm=algorithm, steps=60, hidden=8,
                        learning_starts=4, batch_size=2, replay_capacity=64,
                        train_frequency=2, eval_interval=10, eval_episodes=2,
                        final_episodes=2, eval_seed=76050, final_seed=86000,
                        early_stopping_patience=patience)
    output = tmp_path / algorithm
    training.run(c, output)
    metadata = json.loads((output / "metadata.json").read_text())
    payload = torch.load(output / "model.pt", weights_only=True)
    assert payload["step"] == metadata["actual_steps"] == expected_step
    assert payload["config"]["steps"] == metadata["requested_steps"] == 60
    assert metadata["stop_reason"] == expected_reason
    assert metadata["gradient_updates"] == (expected_step - 4) // 2 + 1
    assert bool(stochastic_calls) == (algorithm == "noisy" and patience > 0)
    assert (output / "COMPLETE").is_file()


def test_patience_resets_and_cap_is_saved_when_no_consecutive_success(tmp_path, monkeypatch):
    scores = iter([500.0, 500.0, 100.0, 500.0])
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: [next(scores)])
    monkeypatch.setattr(training, "evaluate_details", lambda *args, **kwargs: [
        {"seed": 86000, "return": 1.0, "length": 1, "terminated": True, "truncated": False}])
    c = training.Config(steps=30, hidden=8, learning_starts=4, batch_size=2,
                        replay_capacity=64, train_frequency=2, eval_interval=10,
                        eval_episodes=1, final_episodes=1, final_seed=86000,
                        early_stopping_patience=2)
    training.run(c, tmp_path / "cap")
    metadata = json.loads((tmp_path / "cap/metadata.json").read_text())
    assert metadata["actual_steps"] == 30
    assert metadata["stop_reason"] == "budget_exhausted"
    assert metadata["early_stopping"]["consecutive_passes"] == 1
