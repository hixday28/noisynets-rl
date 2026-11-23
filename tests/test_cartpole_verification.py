from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import torch

from noisynets.networks import QNetwork
from noisynets.train import Config, evaluate_details


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_cartpole.py"
SPEC = importlib.util.spec_from_file_location("cartpole_verification", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_incomplete_series_does_not_start_evaluation(tmp_path, monkeypatch):
    def unexpected_evaluation(job):
        raise AssertionError("Незавершённая серия не должна оцениваться")

    monkeypatch.setattr(verifier, "verify_run", unexpected_evaluation)
    output = tmp_path / "audit.csv"
    status = verifier.main(["--family", "dqn", "--results", str(tmp_path), "--output", str(output)])
    summary = json.loads(output.with_suffix(".json").read_text())
    assert status == 1 and not summary["verification_started"]
    assert summary["expected_models"] == 20 and len(summary["pending"]) == 20


def test_a3c_discovery_uses_cartpole_subdirectory(tmp_path):
    folder = tmp_path / "cartpole_a3c_final/cartpole/a3c/seed_400"
    folder.mkdir(parents=True)
    (folder / "COMPLETE").write_text("success\n")
    jobs, pending = verifier.discover_runs(tmp_path, ["a3c"])
    assert jobs == [{"family": "a3c", "condition": "a3c", "seed": 400, "folder": str(folder)}]
    assert len(pending) == 29
    assert all("cartpole_a3c_final/cartpole/" in path for path in pending)


def test_discovery_ignores_sibling_config_and_log_files(tmp_path):
    folder = tmp_path / "cartpole_ppo_final/ppo/seed_400"
    folder.mkdir(parents=True)
    (folder / "COMPLETE").write_text("success\n")
    (folder.parent / "seed_400.json").write_text("{}\n")
    (folder.parent / "seed_400.log").write_text("Готово\n")
    jobs, pending = verifier.discover_runs(tmp_path, ["ppo"])
    assert len(jobs) == 1 and len(pending) == 29
    assert jobs[0]["folder"] == str(folder)


def test_checkpoint_roundtrip_and_corrupted_details_are_detected(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    c = Config(seed=400, hidden=8, final_seed=85000, final_episodes=30)
    model = QNetwork(4, 2, hidden=8)
    records = evaluate_details(model, c.env, c.final_episodes, c.final_seed)
    (tmp_path / "config.json").write_text(json.dumps(asdict(c)))
    details = {"mean_weights": records}
    (tmp_path / "final_episode_details.json").write_text(json.dumps(details))
    (tmp_path / "final_returns.json").write_text(json.dumps({"mean_weights": [row["return"] for row in records]}))
    torch.save({"config": asdict(c), "model": model.state_dict(), "obs_dim": 4, "n_actions": 2,
                "step": c.steps}, tmp_path / "model.pt")
    job = {"family": "dqn", "condition": "dqn", "seed": 400, "folder": str(tmp_path)}
    checked = verifier.verify_run(job)
    assert len(checked) == 1 and checked[0]["matches"] and checked[0]["n_episodes"] == 30

    # Совпадения одних наград недостаточно: seed эпизода тоже проверяется.
    details["mean_weights"][0]["seed"] += 1
    (tmp_path / "final_episode_details.json").write_text(json.dumps(details))
    checked = verifier.verify_run(job)
    assert not checked[0]["matches"] and "seed" in checked[0]["error"]


def test_exact_comparison_detects_terminal_flag_difference_with_same_return():
    saved = [{"seed": 85000, "return": 500.0, "length": 500, "terminated": False, "truncated": True}]
    replayed = [saved[0] | {"terminated": True}]
    difference = json.loads(verifier.first_difference(saved, replayed))
    assert difference["episode_index"] == 0
    assert difference["saved"]["return"] == difference["replayed"]["return"]
    assert difference["saved"]["terminated"] != difference["replayed"]["terminated"]
