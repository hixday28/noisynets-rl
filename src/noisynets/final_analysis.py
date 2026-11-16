"""Анализ основной серии и абляций NoisyNet-DQN."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "dqn": "DQN",
    "noisy": "NoisyNet-DQN",
    "noisy_sigma01": "σ₀ = 0,1",
    "noisy_sigma10": "σ₀ = 1,0",
    "noisy_fixed": "Фиксированный σ",
    "noisy_update": "Шум между обновлениями",
}
COLORS = {"dqn": "#343434", "noisy": "#24649b", "noisy_sigma01": "#638535",
          "noisy_sigma10": "#a45732", "noisy_fixed": "#875986", "noisy_update": "#368787"}
ORDER = list(LABELS)
EXPECTED_SEEDS = set(range(100, 110))
COMMON_CONFIG = ["env", "steps", "hidden", "gamma", "learning_rate", "batch_size",
                 "replay_capacity", "learning_starts", "train_frequency", "target_interval",
                 "epsilon_start", "epsilon_end", "epsilon_decay_steps", "eval_interval",
                 "eval_episodes", "final_episodes", "eval_seed", "final_seed", "gradient_clip"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_groups(manifest: dict) -> dict[str, set[str]]:
    """Список условий из манифеста; структура папок дополнительно проверяется ниже."""
    groups = {}
    environments = manifest.get("environments", {})
    if isinstance(environments, list):
        groups = {spec["key"]: {c["key"] for c in spec["conditions"]} for spec in environments}
    elif isinstance(environments, dict):
        for key, spec in environments.items():
            if isinstance(spec, dict) and "conditions" in spec:
                conditions = spec["conditions"]
                if isinstance(conditions, dict):
                    groups[key] = set(conditions)
                else:
                    groups[key] = {c if isinstance(c, str) else c["name"] for c in conditions}
    for spec in manifest.get("experiments", manifest.get("groups", [])):
        if isinstance(spec, dict) and "condition" in spec:
            key = spec.get("env_key", spec.get("environment"))
            if key:
                groups.setdefault(key, set()).add(spec["condition"])
    return groups


def load_study(root: Path, manifest_path: Path | None = None, strict: bool = True):
    root = Path(root)
    manifest = read_json(manifest_path) if manifest_path is not None else {}
    expected_seeds = set(manifest.get("seeds", sorted(EXPECTED_SEEDS)))
    rows, curves, configs, source_hashes = [], [], {}, []
    prescribed = {}
    for environment in manifest.get("environments", []):
        for condition in environment["conditions"]:
            prescribed[(environment["key"], condition["key"])] = {
                **environment["config"], **{k: v for k, v in condition.items() if k != "key"}}
    for marker in sorted(root.glob("*/*/seed_*/COMPLETE")):
        folder = marker.parent
        env_key, condition = folder.relative_to(root).parts[:2]
        c = read_json(folder / "config.json")
        meta = read_json(folder / "metadata.json")
        returns = read_json(folder / "final_returns.json")
        curve = pd.read_csv(folder / "evaluations.csv")
        if prescribed:
            expected = prescribed.get((env_key, condition))
            if expected is None or any(c.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Параметры не совпадают с манифестом: {env_key}/{condition}/{folder.name}")
        if "training_source_sha256" in meta:
            source_hashes.append(meta["training_source_sha256"])
        if condition not in LABELS:
            raise ValueError(f"Неизвестное условие: {condition}")
        if c["seed"] != int(folder.name.split("_")[-1]):
            raise ValueError(f"Seed папки и конфигурации различаются: {folder}")
        if len(curve) < 2 or curve.step.iloc[0] != 0 or curve.step.iloc[-1] != c["steps"]:
            raise ValueError(f"Неполная кривая обучения: {folder}")
        if not curve.step.is_monotonic_increasing or curve.step.duplicated().any():
            raise ValueError(f"Неверная последовательность оценок: {folder}")
        if not np.isfinite(curve[["step", "return_mean", "return_std"]].to_numpy()).all():
            raise ValueError(f"Нечисловое значение в кривой обучения: {folder}")
        if c["algorithm"] != "dqn":
            columns = [f"{prefix}layers.{layer}" for prefix in ["sigma_abs_", "bias_sigma_abs_"]
                       for layer in [2, 4]]
            if not np.isfinite(curve[columns].to_numpy()).all():
                raise ValueError(f"Нечисловой масштаб шума: {folder}")
        if set(curve.step) != set(range(0, c["steps"] + 1, c["eval_interval"])) | {c["steps"]}:
            raise ValueError(f"Пропущенные точки оценки: {folder}")
        if len(returns["mean_weights"]) != c["final_episodes"]:
            raise ValueError(f"Неверное число итоговых эпизодов: {folder}")
        if c["algorithm"] != "dqn" and len(returns.get("resampled_noise", [])) != c["final_episodes"]:
            raise ValueError(f"Отсутствует стохастическая оценка NoisyNet: {folder}")
        for scores in returns.values():
            if not np.isfinite(np.asarray(scores, dtype=float)).all():
                raise ValueError(f"Нечисловая награда: {folder}")
        scores = np.asarray(returns["mean_weights"], dtype=float)
        noisy_scores = returns.get("resampled_noise", [])
        row = {
            "environment": env_key, "env_id": c["env"], "condition": condition,
            "algorithm": c["algorithm"], "seed": int(c["seed"]), "steps": int(c["steps"]),
            "hidden": int(c["hidden"]),
            "final_mean": float(scores.mean()), "final_episode_sd": float(scores.std(ddof=1)),
            "final_episode_median": float(np.median(scores)),
            "stochastic_mean": float(np.mean(noisy_scores)) if noisy_scores else np.nan,
            "auc_mean": float(np.trapezoid(curve.return_mean, curve.step) / c["steps"]),
            "validation_final": float(curve.return_mean.iloc[-1]),
            "elapsed_seconds": float(meta["elapsed_seconds"]),
            "trainable_parameters": int(meta["trainable_parameters"]),
            "gradient_updates": int(meta["gradient_updates"]),
            "n_final_episodes": int(c["final_episodes"]),
            "sigma0": float(c["sigma0"]), "noise_cadence": c.get("noise_cadence", "action"),
            "success_rate": np.nan, "folder": folder.relative_to(root).as_posix(),
        }
        details_path = folder / "final_episode_details.json"
        if details_path.exists():
            details = read_json(details_path)
            records = (details.get("mean_weights", []) if isinstance(details, dict)
                       else [d for d in details if d.get("mode") == "mean_weights"])
            if records:
                if len(records) != len(scores) or [d["return"] for d in records] != list(scores):
                    raise ValueError(f"Подробности эпизодов не совпадают с наградами: {folder}")
                if [d["seed"] for d in records] != list(range(c["final_seed"], c["final_seed"]+len(scores))):
                    raise ValueError(f"Неверные seed отложенных эпизодов: {folder}")
                if c["env"] == "Acrobot-v1":
                    row["success_rate"] = float(np.mean([bool(d["terminated"]) for d in records]))
                elif c["env"] == "CartPole-v1":
                    row["success_rate"] = float(np.mean([bool(d["truncated"]) for d in records]))
        rows.append(row)
        curves.append(curve.assign(environment=env_key, condition=condition, seed=int(c["seed"])))
        configs[(env_key, condition, int(c["seed"]))] = c
    if not rows:
        raise ValueError(f"Нет завершенных запусков в {root}")
    runs = pd.DataFrame(rows).sort_values(["environment", "condition", "seed"]).reset_index(drop=True)
    if runs.duplicated(["environment", "condition", "seed"]).any():
        raise ValueError("Повторяющиеся запуски")
    if source_hashes and any(h != source_hashes[0] for h in source_hashes):
        raise ValueError("Запуски выполнены разными версиями исходного кода")
    observed_groups = {}
    for (env_key, condition), group in runs.groupby(["environment", "condition"]):
        observed_groups.setdefault(env_key, set()).add(condition)
        if strict and set(group.seed) != expected_seeds:
            raise ValueError(f"Неполный набор seed: {env_key}/{condition}")
    declared = expected_groups(manifest)
    if strict and declared and observed_groups != declared:
        raise ValueError(f"Условия не совпадают с манифестом: {observed_groups} / {declared}")
    for env_key, group in runs.groupby("environment"):
        if strict and not {"dqn", "noisy"}.issubset(set(group.condition)):
            raise ValueError(f"Нет основной пары методов: {env_key}")
        baseline = configs[(env_key, group.condition.iloc[0], int(group.seed.iloc[0]))]
        for _, row in group.iterrows():
            c = configs[(env_key, row.condition, int(row.seed))]
            if any(c[k] != baseline[k] for k in COMMON_CONFIG):
                raise ValueError(f"Различающиеся общие параметры: {env_key}/{row.condition}")
    return runs, pd.concat(curves, ignore_index=True), manifest


def paired_interval(a: pd.Series, b: pd.Series, seed: int = 7241, samples: int = 20000):
    a, b = a.sort_index(), b.sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError("Для парного сравнения необходимы совпадающие уникальные seed")
    delta = (a - b).to_numpy(dtype=float)
    if len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("Для интервала нужны как минимум две конечные парные разности")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(delta, (samples, len(delta)), replace=True).mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {"mean_difference": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "n_pairs": len(delta), "paired": True, "resamples": samples,
            "differences_by_seed": {str(k): float(v) for k, v in zip(a.index, delta)}}


def summarize_study(runs: pd.DataFrame) -> dict:
    conditions, comparisons, ablations, modes, stochastic_comparisons = [], [], [], [], []
    for (env, condition), group in runs.groupby(["environment", "condition"], sort=True):
        conditions.append({
            "environment": env, "env_id": group.env_id.iloc[0], "condition": condition,
            "label": LABELS[condition], "n_seeds": len(group), "steps": int(group.steps.iloc[0]),
            "total_training_steps": int(group.steps.sum()),
            "final_mean": float(group.final_mean.mean()),
            "final_seed_sd": float(group.final_mean.std(ddof=1)),
            "final_seed_median": float(group.final_mean.median()),
            "final_min": float(group.final_mean.min()), "final_max": float(group.final_mean.max()),
            "stochastic_mean": float(group.stochastic_mean.mean()) if group.stochastic_mean.notna().any() else None,
            "stochastic_seed_sd": float(group.stochastic_mean.std(ddof=1)) if group.stochastic_mean.notna().any() else None,
            "auc_mean": float(group.auc_mean.mean()), "auc_seed_sd": float(group.auc_mean.std(ddof=1)),
            "success_rate": float(group.success_rate.mean()) if group.success_rate.notna().any() else None,
            "seconds_mean": float(group.elapsed_seconds.mean()),
            "trainable_parameters": int(group.trainable_parameters.iloc[0]),
        })
    for env, group in runs.groupby("environment", sort=True):
        for condition in ORDER:
            if condition == "dqn" or condition not in set(group.condition):
                continue
            for metric in ["final_mean", "auc_mean"]:
                a = group.loc[group.condition == condition].set_index("seed")[metric]
                b = group.loc[group.condition == "dqn"].set_index("seed")[metric]
                if not b.empty:
                    comparisons.append({"environment": env, "condition": condition, "reference": "dqn",
                                        "comparison_type": "primary" if condition == "noisy" else "exploratory",
                                        "metric": metric, **paired_interval(a, b)})
                if condition != "noisy":
                    b = group.loc[group.condition == "noisy"].set_index("seed")[metric]
                    if not b.empty:
                        ablations.append({"environment": env, "condition": condition, "reference": "noisy",
                                          "metric": metric, **paired_interval(a, b)})
            data = group.loc[group.condition == condition].set_index("seed")
            if data.stochastic_mean.notna().all():
                modes.append({"environment": env, "condition": condition,
                              "metric": "stochastic_minus_mean_weights",
                              **paired_interval(data.stochastic_mean, data.final_mean)})
            if condition == "noisy":
                baseline = group.loc[group.condition == "dqn"].set_index("seed").final_mean
                stochastic_comparisons.append({
                    "environment": env, "condition": condition, "reference": "dqn",
                    "metric": "stochastic_mean", "comparison_type": "post_hoc_exploratory",
                    **paired_interval(data.stochastic_mean, baseline)})
    return {"conditions": conditions, "comparisons_vs_dqn": comparisons,
            "ablations_vs_noisy": ablations, "evaluation_mode_differences": modes,
            "stochastic_vs_dqn": stochastic_comparisons,
            "n_training_runs": len(runs),
            "total_training_steps": int(runs.steps.sum()),
            "total_heldout_episodes_mean_weights": int(runs.n_final_episodes.sum()),
            "replication_unit": "seed обучения; для разностей сохраняется пара методов с общей инициализацией",
            "interval_method": "процентильный парный bootstrap, 20 000 перевыборок, seed 7241",
            "multiple_comparisons": "основные сравнения заданы до серии; все интервалы описательные, без поправки на множественные сравнения",
            "evaluation": "основная оценка по средним весам; стохастическая оценка NoisyNet анализируется отдельно"}


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_study_figures(runs: pd.DataFrame, curves: pd.DataFrame, summary: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.dpi": 220})
    for env, group in runs.groupby("environment", sort=True):
        env_id = group.env_id.iloc[0]
        bounds = (0, 510) if env_id == "CartPole-v1" else (-510, 10)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        learning_bounds = list(bounds)
        for condition in ["dqn", "noisy"]:
            data = curves.loc[(curves.environment == env) & (curves.condition == condition)]
            if data.empty:
                continue
            stat = data.groupby("step").return_mean.agg(["mean", "std"])
            ax.plot(stat.index, stat["mean"], color=COLORS[condition], label=LABELS[condition], lw=2)
            ax.fill_between(stat.index, stat["mean"]-stat["std"], stat["mean"]+stat["std"],
                            color=COLORS[condition], alpha=.14)
            learning_bounds[0] = min(learning_bounds[0], float((stat["mean"]-stat["std"]).min())-10)
            learning_bounds[1] = max(learning_bounds[1], float((stat["mean"]+stat["std"]).max())+10)
        ax.set(xlabel="Шаги взаимодействия со средой", ylabel="Средняя награда за эпизод",
               title=env_id, ylim=learning_bounds)
        if env_id == "Acrobot-v1":
            ax.axhline(-500, color="#888888", lw=.8, ls=":")
        ax.grid(alpha=.2); ax.legend(frameon=False)
        _finish(fig, output/f"{env}_learning.png")

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for condition, shift in [("dqn", -.07), ("noisy", .07)]:
            data = group.loc[group.condition == condition]
            ax.scatter(data.seed+shift, data.final_mean, label=LABELS[condition],
                       color=COLORS[condition], s=35, zorder=3)
        ax.set(xlabel="Seed обучения", ylabel="Награда на отложенных эпизодах", title=env_id,
               xticks=sorted(group.seed.unique()), ylim=bounds)
        ax.grid(alpha=.2); ax.legend(frameon=False)
        _finish(fig, output/f"{env}_final_by_seed.png")

        noisy = group.loc[group.condition == "noisy"].sort_values("seed")
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        x = np.arange(len(noisy)); width=.36
        ax.bar(x-width/2, noisy.final_mean, width, color="#343434", label="Средние веса μ")
        ax.bar(x+width/2, noisy.stochastic_mean, width, color="#24649b", label="С включённым шумом")
        ax.set(xticks=x, xticklabels=noisy.seed, xlabel="Seed обучения",
               ylabel="Награда на отложенных эпизодах", title=env_id, ylim=bounds)
        ax.set_title(env_id, pad=38)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=9)
        ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
        _finish(fig, output/f"{env}_evaluation_modes.png")

        environment = gym.make(env_id)
        try:
            n_actions = int(environment.action_space.n)
        finally:
            environment.close()
        hidden, sigma0 = int(noisy.hidden.iloc[0]), float(noisy.sigma0.iloc[0])
        noise_curve = curves.loc[(curves.environment == env) & (curves.condition == "noisy")]
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
        for ax, prefix, title in zip(axes, ["sigma_abs_", "bias_sigma_abs_"], ["Веса", "Смещения"]):
            for suffix, label, in_features, out_features in [
                ("layers.2", "Второй скрытый слой", hidden, hidden),
                ("layers.4", "Выходной слой", hidden, n_actions),
            ]:
                initial = sigma0 / np.sqrt(in_features if prefix == "sigma_abs_" else out_features)
                stat = noise_curve.groupby("step")[prefix+suffix].agg(["mean", "std"])
                ax.plot(stat.index, stat["mean"], label=f"{label}: σ(0) = {initial:.4f}", lw=2)
                ax.fill_between(stat.index, stat["mean"]-stat["std"], stat["mean"]+stat["std"], alpha=.13)
            ax.set(ylabel="Средний модуль σ", title=title)
            ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
        axes[-1].set_xlabel("Шаги взаимодействия со средой")
        fig.suptitle(f"{env_id}, σ₀ = {sigma0:g}")
        _finish(fig, output/f"{env}_sigma.png")

        if len(set(group.condition)) > 2:
            conditions = [c for c in ORDER if c in set(group.condition)]
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            ablation_bounds = list(bounds)
            for i, condition in enumerate(conditions):
                data = group.loc[group.condition == condition].sort_values("seed")
                offsets = np.linspace(-.15,.15,len(data))
                ax.scatter(np.full(len(data),i)+offsets, data.final_mean,
                           s=25, color=COLORS[condition], alpha=.7)
                ax.errorbar(i, data.final_mean.mean(), yerr=data.final_mean.std(ddof=1),
                            fmt="D", color="black", ms=5, capsize=4, zorder=5)
                mean, sd = data.final_mean.mean(), data.final_mean.std(ddof=1)
                ablation_bounds[0] = min(ablation_bounds[0], mean-sd-10)
                ablation_bounds[1] = max(ablation_bounds[1], mean+sd+10)
            labels = [LABELS[c].replace("Шум между обновлениями", "Шум между\nобновлениями")
                      .replace("Фиксированный σ", "Фиксированный\nσ") for c in conditions]
            ax.set(xticks=range(len(conditions)), xticklabels=labels, ylabel="Награда на отложенных эпизодах",
                   ylim=ablation_bounds, title=env_id)
            if env_id == "Acrobot-v1":
                ax.axhline(-500, color="#888888", lw=.8, ls=":")
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
            _finish(fig, output/f"{env}_ablations.png")

            records = [c for c in summary["ablations_vs_noisy"]
                       if c["environment"] == env and c["metric"] == "final_mean"]
            fig, ax = plt.subplots(figsize=(7.2, 3.5))
            for i, record in enumerate(records):
                mid=record["mean_difference"]; lo=record["ci95_low"]; hi=record["ci95_high"]
                ax.errorbar(mid,i,xerr=np.array([[mid-lo],[hi-mid]]),fmt="o",
                            color=COLORS[record["condition"]],capsize=4)
            ax.axvline(0,color="#666666",lw=1,ls="--")
            ax.set(yticks=range(len(records)), yticklabels=[LABELS[c["condition"]] for c in records],
                   xlabel="Вариант − основной NoisyNet-DQN\nПарный 95%-интервал разности награды", title=env_id)
            ax.grid(axis="x",alpha=.2)
            _finish(fig, output/f"{env}_ablation_differences.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metric, title in zip(axes, ["final_mean", "auc_mean"],
                                 ["Отложенная награда", "AUC / T"]):
        records = [c for c in summary["comparisons_vs_dqn"]
                   if c["condition"] == "noisy" and c["metric"] == metric]
        for i, record in enumerate(records):
            mid, lo, hi = [record[k] for k in ["mean_difference", "ci95_low", "ci95_high"]]
            ax.errorbar(mid, i, xerr=np.array([[mid-lo], [hi-mid]]),
                        fmt="o", color=COLORS["noisy"], capsize=4)
        names = [runs.loc[runs.environment == c["environment"], "env_id"].iloc[0] for c in records]
        ax.set(yticks=range(len(records)), yticklabels=names, title=title,
               xlabel="Разность Noisy − DQN")
        ax.set_ylim(-.7, len(records)-.3)
        ax.axvline(0, color="#666666", lw=1, ls="--")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Основные сравнения: средняя разность и парный 95%-интервал", fontsize=10)
    _finish(fig, output/"main_differences.png")


def verify_checkpoints(root: Path):
    import torch
    from .evaluate import load_checkpoint
    from .train import evaluate_details
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records=[]
    for marker in sorted(Path(root).glob("*/*/seed_*/COMPLETE")):
        folder=marker.parent; env,condition=folder.relative_to(root).parts[:2]
        model,c=load_checkpoint(folder/"model.pt")
        if c != read_json(folder/"config.json"):
            raise AssertionError(f"Конфигурация модели не совпадает: {env}/{condition}/{c['seed']}")
        saved=read_json(folder/"final_returns.json")
        details_path=folder/"final_episode_details.json"
        saved_details=read_json(details_path) if details_path.exists() else None
        for mode,stochastic in [("mean_weights",False),("resampled_noise",True)]:
            if mode not in saved:
                continue
            details=evaluate_details(model,c["env"],c["final_episodes"],c["final_seed"],stochastic=stochastic)
            values=[record["return"] for record in details]
            if values != saved[mode]:
                raise AssertionError(f"Награды модели не совпадают: {env}/{condition}/{c['seed']}/{mode}")
            if saved_details is not None and details != saved_details[mode]:
                raise AssertionError(f"Подробности эпизодов не совпадают: {env}/{condition}/{c['seed']}/{mode}")
            records.append({"environment":env,"condition":condition,"seed":c["seed"],"mode":mode,
                            "n_episodes":len(values),"mean_return":float(np.mean(values)),"matches":True})
    return pd.DataFrame(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/final"))
    parser.add_argument("--manifest",type=Path,default=Path("configs/final_study.json"))
    parser.add_argument("--out",type=Path,default=Path("results/final_analysis"))
    args=parser.parse_args()
    runs,curves,manifest=load_study(args.results,args.manifest)
    summary=summarize_study(runs)
    args.out.mkdir(parents=True,exist_ok=True)
    runs.to_csv(args.out/"per_run.csv",index=False)
    curves.to_csv(args.out/"curves.csv",index=False)
    pd.DataFrame(summary["conditions"]).to_csv(args.out/"conditions.csv",index=False)
    pd.DataFrame(summary["comparisons_vs_dqn"]).to_csv(args.out/"comparisons.csv",index=False)
    pd.DataFrame(summary["ablations_vs_noisy"]).to_csv(args.out/"ablations.csv",index=False)
    pd.DataFrame(summary["stochastic_vs_dqn"]).to_csv(args.out/"stochastic_comparisons.csv",index=False)
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    make_study_figures(runs,curves,summary,args.out/"figures")
    print(json.dumps({"n_runs":len(runs),"training_steps":summary["total_training_steps"],
                      "conditions":summary["conditions"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
