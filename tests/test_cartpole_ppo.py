from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from noisynets.actorcritic import evaluate_policy
from noisynets.cartpole_ppo import CartPolePPOConfig, SeparateActorCritic
from noisynets.evaluate_policy import load_checkpoint


CONDITIONS = [("ppo", "factorized"), ("noisy-ppo", "factorized"), ("noisy-ppo", "independent")]


def test_mean_initialization_is_identical_for_all_conditions():
    torch.set_num_threads(1)
    models = []
    for algorithm, generation in CONDITIONS:
        torch.manual_seed(31)
        models.append(SeparateActorCritic(4, 2, CartPolePPOConfig(
            algorithm=algorithm, generation=generation, hidden=16, sigma0=.01)))
    for branch in ["actor", "critic"]:
        for index in [0, 2, 4]:
            baseline = getattr(models[0], branch)[index]
            for model in models[1:]:
                layer = getattr(model, branch)[index]
                weight = layer.weight_mu if hasattr(layer, "weight_mu") else layer.weight
                bias = layer.bias_mu if hasattr(layer, "bias_mu") else layer.bias
                assert torch.equal(baseline.weight, weight)
                assert torch.equal(baseline.bias, bias)
    assert models[0].actor[0].weight.data_ptr() != models[0].critic[0].weight.data_ptr()
    final_actor = models[0].actor[4].weight.detach()
    assert torch.allclose(final_actor @ final_actor.T, torch.eye(2)*.01**2, atol=1e-8)


@pytest.mark.parametrize("algorithm,generation", CONDITIONS)
def test_checkpoint_roundtrip_preserves_all_evaluation_modes(tmp_path, algorithm, generation):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    c = CartPolePPOConfig(algorithm=algorithm, generation=generation, hidden=8,
                          sigma0=.01, final_seed=85000, final_episodes=2)
    model = SeparateActorCritic(4, 2, c)
    model.reset_noise(torch.Generator().manual_seed(9))
    path = tmp_path / "model.pt"
    torch.save({"config": asdict(c), "model": model.state_dict(), "obs_dim": 4,
                "n_actions": 2, "step": c.steps}, path)
    restored, config = load_checkpoint(path)
    assert asdict(config) == asdict(c)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    modes = ["behaviour", "mean_greedy", "mean_sampled"] if c.noisy else ["behaviour", "mean_greedy"]
    for mode in modes:
        assert evaluate_policy(restored, config, 2, 85000, mode) == evaluate_policy(model, c, 2, 85000, mode)


@pytest.mark.parametrize("complete", [False, True])
def test_final_runner_does_not_overwrite_existing_runs(tmp_path, complete):
    output = tmp_path / "separate_output"
    folder = output / "ppo/seed_400"
    folder.mkdir(parents=True)
    original = json.dumps({"algorithm": "noisy-ppo", "seed": 400})
    (folder / "config.json").write_text(original)
    if complete:
        (folder / "COMPLETE").write_text("success\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"config": {"algorithm": "ppo"},
                                    "conditions": [{"key": "ppo", "algorithm": "ppo"}],
                                    "seeds": [400], "trainer_module": "noisynets.cartpole_ppo",
                                    "results_root": "unused"}))
    script = Path(__file__).resolve().parents[1] / "scripts/cartpole_ppo_final.py"
    result = subprocess.run([sys.executable, str(script), "--manifest", str(manifest),
                             "--out", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert (folder / "config.json").read_text() == original
    assert not (folder / "model.pt").exists()
    assert ("параметры отличаются" in result.stderr if complete else "Незавершённый запуск" in result.stderr)
