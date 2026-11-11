"""Сводные результаты по независимым запускам обучения."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"dqn": "DQN", "noisy": "NoisyNet-DQN", "noisy-fixed": "NoisyNet fixed σ"}
COLORS = {"dqn": "#333333", "noisy": "#30659b", "noisy-fixed": "#7c4762"}


def load_runs(root: Path):
    rows, curves = [], []
    for marker in sorted(root.glob("*/seed_*/COMPLETE")):
        folder = marker.parent
        c = json.loads((folder / "config.json").read_text())
        meta = json.loads((folder / "metadata.json").read_text())
        final = json.loads((folder / "final_returns.json").read_text())
        curve = pd.read_csv(folder / "evaluations.csv")
        if curve.step.iloc[-1] != c["steps"] or len(final["mean_weights"]) != c["final_episodes"]:
            raise ValueError(f"Incomplete run: {folder}")
        row = {"algorithm": c["algorithm"], "seed": c["seed"], "steps": c["steps"],
               "final_mean": np.mean(final["mean_weights"]),
               "final_episode_sd": np.std(final["mean_weights"], ddof=1),
               "validation_final": curve.return_mean.iloc[-1],
               "auc_mean": np.trapezoid(curve.return_mean, curve.step) / c["steps"],
               "noisy_policy_mean": np.mean(final.get("resampled_noise", [np.nan])),
               "elapsed_seconds": meta["elapsed_seconds"],
               "trainable_parameters": meta["trainable_parameters"]}
        rows.append(row)
        curves.append(curve.assign(algorithm=c["algorithm"], seed=c["seed"]))
    if not rows:
        raise ValueError(f"No completed runs under {root}")
    configs = [json.loads((m.parent / "config.json").read_text())
               for m in sorted(root.glob("*/seed_*/COMPLETE"))]
    common_keys = [k for k in configs[0] if k not in {"algorithm", "seed"}]
    if any(any(c[k] != configs[0][k] for k in common_keys) for c in configs[1:]):
        raise ValueError("Aggregate only runs with the same experimental protocol")
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def summarise(runs: pd.DataFrame):
    rows = []
    for alg, data in runs.groupby("algorithm", sort=True):
        rows.append({"algorithm": alg, "n_seeds": len(data),
                     "final_mean": float(data.final_mean.mean()),
                     "final_seed_sd": float(data.final_mean.std(ddof=1)),
                     "auc_mean": float(data.auc_mean.mean()),
                     "auc_seed_sd": float(data.auc_mean.std(ddof=1)),
                     "final_noisy_mean": float(data.noisy_policy_mean.mean())
                     if data.noisy_policy_mean.notna().any() else None,
                     "seconds_mean": float(data.elapsed_seconds.mean())})
    result = {"algorithms": rows, "unit_of_replication": "independent training seed"}
    if {"dqn", "noisy"}.issubset(set(runs.algorithm)):
        a = runs.loc[runs.algorithm == "noisy"].set_index("seed").final_mean.sort_index()
        b = runs.loc[runs.algorithm == "dqn"].set_index("seed").final_mean.sort_index()
        rng = np.random.default_rng(7241)
        paired = a.index.is_unique and b.index.is_unique and a.index.equals(b.index)
        if paired:
            # При одинаковых seed сохраняем пары инициализаций при перевыборке.
            delta = (a - b).to_numpy()
            differences = rng.choice(delta, (20000, len(delta)), replace=True).mean(axis=1)
        else:
            differences = (rng.choice(a.to_numpy(), (20000, len(a)), replace=True).mean(axis=1)
                           - rng.choice(b.to_numpy(), (20000, len(b)), replace=True).mean(axis=1))
        result["difference_noisy_minus_dqn"] = {
            "mean": float(a.mean() - b.mean()),
            "bootstrap_percentile_95": [float(v) for v in np.quantile(differences, [0.025, 0.975])],
            "resamples": 20000, "paired": paired,
            "caveat": "Исследовательский интервал при малом числе запусков обучения"}
    return result


def make_figures(runs: pd.DataFrame, curves: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.dpi": 220})
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    for alg, data in curves.groupby("algorithm", sort=True):
        stat = data.groupby("step").return_mean.agg(["mean", "std"])
        ax.plot(stat.index, stat["mean"], label=LABELS[alg], color=COLORS[alg], lw=2)
        ax.fill_between(stat.index, stat["mean"]-stat["std"], stat["mean"]+stat["std"],
                        alpha=0.14, color=COLORS[alg])
    ax.set(xlabel="Шаги взаимодействия со средой", ylabel="Средняя награда за эпизод", ylim=(0, 510))
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "learning_curves.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    for alg, data in runs.groupby("algorithm", sort=True):
        ax.plot(data.seed, data.final_mean, "o", label=LABELS[alg], color=COLORS[alg], ms=6)
    ax.set(xlabel="Seed обучения", ylabel="Награда на отложенных эпизодах", ylim=(0, 510),
           xticks=sorted(runs.seed.unique()))
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "final_by_seed.png"); plt.close(fig)

    noisy = curves.loc[curves.algorithm == "noisy"]
    if not noisy.empty:
        fig, ax = plt.subplots(figsize=(7.1, 3.6))
        for col, label in [("sigma_abs_layers.2", "Скрытый слой 64 → 64"),
                           ("sigma_abs_layers.4", "Выходной слой 64 → 2")]:
            stat = noisy.groupby("step")[col].agg(["mean", "std"])
            ax.plot(stat.index, stat["mean"], label=label, lw=2)
            ax.fill_between(stat.index, stat["mean"]-stat["std"], stat["mean"]+stat["std"], alpha=0.13)
        ax.set(xlabel="Шаги взаимодействия со средой", ylabel="Среднее |σ| весов")
        ax.grid(alpha=0.2); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output / "sigma_evolution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    data = runs.loc[runs.algorithm == "noisy"]
    if not data.empty:
        x = np.arange(len(data)); width=0.34
        ax.bar(x-width/2, data.final_mean, width, label="Средние веса μ", color="#444444")
        ax.bar(x+width/2, data.noisy_policy_mean, width, label="Шум при выборе действий", color="#30659b")
        ax.set(xticks=x, xticklabels=data.seed, xlabel="Seed обучения",
               ylabel="Награда на отложенных эпизодах", ylim=(0,510))
        ax.legend(frameon=False); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(output / "evaluation_modes.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/pilot"))
    p.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs, curves = load_runs(args.results)
    summary = summarise(runs)
    runs.to_csv(args.out / "per_run.csv", index=False)
    curves.to_csv(args.out / "curves.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    make_figures(runs, curves, args.out / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
