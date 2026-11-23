"""Сопоставление исходных и повторных запусков CartPole-v1."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 475
CONDITIONS = (
    ("DQN", "DQN", "dqn", "mean_weights"),
    ("Noisy-DQN", "DQN", "noisy", "mean_weights"),
    ("A3C", "A3C", "a3c", "behaviour"),
    ("Noisy-A3C, фактор.", "A3C", "noisy_a3c_factorized", "behaviour"),
    ("Noisy-A3C, незав.", "A3C", "noisy_a3c_independent", "behaviour"),
    ("PPO", "PPO", "ppo", "behaviour"),
    ("Noisy-PPO, фактор.", "PPO", "noisy_ppo_factorized", "behaviour"),
    ("Noisy-PPO, незав.", "PPO", "noisy_ppo_independent", "behaviour"),
)
MANIFESTS = {
    "DQN": "cartpole_dqn_final.json",
    "A3C": "cartpole_a3c_final_manifest.json",
    "PPO": "cartpole_ppo_final.json",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_dir(root, family, condition, updated):
    if updated:
        series = root / f"cartpole_{family.lower()}_final"
        return (series / "cartpole" if family in {"DQN", "A3C"} else series) / condition
    return root / ("final" if family == "DQN" else "policy") / "cartpole" / condition


def read_run(folder, seed, final_seed, primary):
    config = read_json(folder / "config.json")
    assert (folder / "COMPLETE").is_file(), folder
    assert config["env"] == "CartPole-v1" and config["seed"] == seed, folder
    assert config["final_seed"] == final_seed and config["final_episodes"] == 30, folder
    with (folder / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in history]
    reward = np.array([float(row["return_mean"]) for row in history])
    expected_steps = list(range(0, config["steps"], config["eval_interval"])) + [config["steps"]]
    assert steps == expected_steps, folder
    assert np.all((reward >= 1) & (reward <= 500)), folder
    evaluations = read_json(folder / "final_episode_details.json")
    returns = read_json(folder / "final_returns.json")
    assert primary in evaluations and set(evaluations) == set(returns), folder
    for mode, episodes in evaluations.items():
        assert [e["seed"] for e in episodes] == list(range(final_seed, final_seed + 30)), folder
        assert [e["return"] for e in episodes] == returns[mode], folder
        for e in episodes:
            assert 1 <= e["return"] == e["length"] <= 500, folder
            assert (e["terminated"] or e["truncated"]) and (not e["truncated"] or e["length"] == 500), folder
    return config, steps, reward, returns


def planned_config(family, condition):
    manifest = read_json(ROOT / "configs" / MANIFESTS[family])
    assert manifest["seeds"] == list(range(400, 410)), family
    environment = manifest["environments"][0] if "environments" in manifest else manifest
    selected = next(row for row in environment["conditions"] if row["key"] == condition)
    return environment["config"] | {key: value for key, value in selected.items() if key != "key"}


def summarize(root):
    groups, runs, curves = [], [], {}
    for label, family, condition, primary in CONDITIONS:
        for updated in (False, True):
            base = source_dir(root, family, condition, updated)
            seeds = range(400, 410) if updated else range(100, 110) if family == "DQN" else range(200, 210)
            final_seed = 85000 if updated else 40000 if family == "DQN" else 65000
            planned = planned_config(family, condition) if updated else {}
            assert {p.name for p in base.glob("seed_*") if p.is_dir()} == {f"seed_{s}" for s in seeds}, base
            model_returns, steps_reached, auc, config_ref, histories = {}, [], [], None, []
            for seed in seeds:
                config, steps, reward, returns = read_run(base / f"seed_{seed}", seed, final_seed, primary)
                settings = {key: value for key, value in config.items() if key != "seed"}
                assert config_ref is None or settings == config_ref, base
                assert all(settings.get(key) == value for key, value in planned.items()), base
                config_ref = settings
                reached = next((step for step, value in zip(steps, reward) if value >= THRESHOLD), None)
                steps_reached.append(reached)
                auc.append(float(np.trapezoid(reward, steps) / steps[-1]))
                histories.append(reward)
                for mode, scores in sorted(returns.items()):
                    score = float(np.mean(scores))
                    model_returns.setdefault(mode, []).append(score)
                    runs.append({
                        "series": "tuned" if updated else "original", "model": label, "seed": seed,
                        "mode": mode, "return_mean": score, "validation_max": float(reward.max()),
                        "validation_last": float(reward[-1]), "first_threshold_step": reached,
                        "episodes_at_limit": scores.count(500),
                    })
            group = {
                "model": label, "family": family, "condition": condition, "updated": updated,
                "primary_mode": primary, "config": config_ref,
                "validation_reached": sum(step is not None for step in steps_reached),
                "median_first_step": float(np.median([s for s in steps_reached if s is not None]))
                if any(s is not None for s in steps_reached) else None,
                "validation_final_reached": sum(int(h[-1] >= THRESHOLD) for h in histories),
                "validation_auc": float(np.mean(auc)),
                "modes": {
                    mode: {
                        "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)),
                        "min": float(min(scores)), "solved": sum(s >= THRESHOLD for s in scores),
                    }
                    for mode, scores in model_returns.items()
                },
            }
            groups.append(group)
            curves[label, updated] = (np.array(steps), np.stack(histories))
    return groups, runs, curves


def tuning_results(root):
    rows = []
    for series in sorted(root.glob("cartpole_*_tuning*")):
        for marker in sorted(series.rglob("COMPLETE")):
            folder = marker.parent
            config = read_json(folder / "config.json")
            assert 300 <= config["seed"] < 400 and config["final_seed"] < 85000, folder
            for mode, scores in sorted(read_json(folder / "final_returns.json").items()):
                rows.append({
                    "series": series.name, "condition": str(folder.parent.relative_to(series)),
                    "algorithm": config["algorithm"], "seed": config["seed"],
                    "steps": config["steps"], "mode": mode, "return_mean": float(np.mean(scores)),
                })
    return rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(groups, curves, path):
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.2), squeeze=False, layout="constrained")
    colors = ("#363636", "#1f6699", "#a76721")
    styles = ("-", "--", ":")
    for family, axis in zip(("DQN", "A3C", "PPO"), axes[:, 0]):
        axis.axhline(THRESHOLD, linestyle=":", color="#777777", linewidth=1.1, label="Порог 475")
        conditions = [item for item in groups if item["updated"] and item["family"] == family]
        for item, color, style in zip(conditions, colors, styles):
            steps, histories = curves[item["model"], True]
            mean = histories.mean(axis=0)
            sd = histories.std(axis=0, ddof=1)
            axis.plot(steps / 1000, mean, color=color, linestyle=style, linewidth=1.8, label=item["model"])
            axis.fill_between(steps / 1000, mean - sd, mean + sd, color=color, alpha=.13)
        axis.set_title(family, fontsize=13)
        axis.set_ylabel("Возврат", fontsize=12)
        axis.grid(alpha=.2)
        axis.legend(loc="lower right", fontsize=10, ncols=2)
    axes[-1, 0].set_xlabel("Шаги среды, тыс.", fontsize=12)
    fig.savefig(path, dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results/cartpole_analysis")
    args = parser.parse_args()
    groups, runs, curves = summarize(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"threshold": THRESHOLD, "groups": groups}, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_rows(args.out / "runs.csv", runs)
    tuning = tuning_results(args.results)
    if tuning:
        write_rows(args.out / "tuning.csv", tuning)
    plot(groups, curves, args.out / "learning.png")
    for group in groups:
        score = group["modes"][group["primary_mode"]]
        print(f'{group["model"]}, {"повтор" if group["updated"] else "исходный"}: '
              f'{score["mean"]:.2f} ± {score["sd"]:.2f}, порог {score["solved"]}/10')


if __name__ == "__main__":
    main()
