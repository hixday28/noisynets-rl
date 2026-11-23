"""Независимая сводка проспективной серии с остановкой по валидации."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def first_validation_stop(curve, config):
    consecutive = 0
    for row in curve:
        if int(row["step"]) == 0:
            continue
        passed = float(row["return_mean"]) >= config["early_stopping_threshold"]
        if config["algorithm"] != "dqn":
            passed = passed and float(row["stochastic_return_mean"]) >= config["early_stopping_threshold"]
        consecutive = consecutive + 1 if passed else 0
        if consecutive >= config["early_stopping_patience"]:
            return int(row["step"]), consecutive
    return None, consecutive


def summarize_values(values):
    a = np.asarray(values, dtype=float)
    require(np.isfinite(a).all(), "Нечисловой результат")
    return {"mean": float(a.mean()), "sample_sd": float(a.std(ddof=1)),
            "median": float(np.median(a)), "minimum": float(a.min()), "maximum": float(a.max())}


def inspect_run(folder, expected, freeze):
    import torch

    config = read_json(folder / "config.json")
    require(all(config.get(k) == value for k, value in expected.items()), f"Конфигурация: {folder}")
    require(config["eval_seed"] == 76050 and config["eval_episodes"] == 10, f"Validation starts: {folder}")
    require(config["final_seed"] == 86000 and config["final_episodes"] == 30, f"Holdout starts: {folder}")
    require(config["early_stopping_patience"] == 2 and config["early_stopping_threshold"] == 475.0,
            f"Правило остановки: {folder}")
    metadata = read_json(folder / "metadata.json")
    require(metadata["training_source_sha256"] == freeze["training_source_sha256"], f"Исходники: {folder}")
    require(datetime.fromisoformat(metadata["started_utc"]) >= datetime.fromisoformat(freeze["frozen_utc"]),
            f"Протокол зафиксирован после начала обучения: {folder}")
    with (folder / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        curve = list(csv.DictReader(stream))
    actual, cap = metadata["actual_steps"], config["steps"]
    expected_steps = sorted(set(range(0, actual + 1, config["eval_interval"])) | {actual})
    require([int(row["step"]) for row in curve] == expected_steps, f"Расписание оценки: {folder}")
    require(all(np.isfinite(float(value)) for row in curve for value in row.values()), f"NaN/Inf в curve: {folder}")
    require(0 < actual <= cap == metadata["requested_steps"], f"Фактический бюджет: {folder}")
    trigger, consecutive = first_validation_stop(curve, config)
    if trigger is None:
        require(actual == cap and metadata["stop_reason"] == "budget_exhausted", f"Неверный cap: {folder}")
    else:
        require(actual == trigger and metadata["stop_reason"] == "validation_threshold",
                f"Сохранён не первый допустимый checkpoint: {folder}")
    require(metadata["early_stopping"]["consecutive_passes"] == consecutive, f"Счётчик patience: {folder}")
    first_update = ((config["learning_starts"] + config["train_frequency"] - 1)
                    // config["train_frequency"]) * config["train_frequency"]
    updates = max(0, (actual - first_update) // config["train_frequency"] + 1)
    require(metadata["gradient_updates"] == updates, f"Число обновлений: {folder}")
    payload = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    require(payload["step"] == actual and payload["config"] == config, f"Шаг/конфигурация checkpoint: {folder}")
    require(all(torch.isfinite(value).all().item() for value in payload["model"].values()), f"NaN/Inf в модели: {folder}")
    details, returns = read_json(folder / "final_episode_details.json"), read_json(folder / "final_returns.json")
    modes = {"mean_weights"} if config["algorithm"] == "dqn" else {"mean_weights", "resampled_noise"}
    require(set(details) == set(returns) == modes, f"Режимы оценки: {folder}")
    means = {}
    for mode in modes:
        records = details[mode]
        require(len(records) == len(returns[mode]) == 30, f"Число итоговых эпизодов: {folder}/{mode}")
        require([row["seed"] for row in records] == list(range(86000, 86030)), f"Holdout seed: {folder}/{mode}")
        require([row["return"] for row in records] == returns[mode], f"Details/returns: {folder}/{mode}")
        for row in records:
            require(1 <= row["length"] == row["return"] <= 500, f"Награда/длина: {folder}/{mode}")
            require(row["terminated"] or row["truncated"], f"Признак завершения: {folder}/{mode}")
            require(row["truncated"] == (row["length"] == 500), f"TimeLimit: {folder}/{mode}")
        means[mode] = float(np.mean(returns[mode]))
    return {"condition": config["algorithm"], "seed": config["seed"], "actual_steps": actual,
            "step_cap": cap, "stop_reason": metadata["stop_reason"], "stopped_before_cap": actual < cap,
            "gradient_updates": updates, "final_validation_mean": float(curve[-1]["return_mean"]),
            "final_validation_stochastic_mean": float(curve[-1]["stochastic_return_mean"])
            if config["algorithm"] != "dqn" else None,
            "mean_weights": means["mean_weights"], "resampled_noise": means.get("resampled_noise"),
            "all_modes_ge475": all(value >= 475 for value in means.values()),
            "folder": folder.relative_to(ROOT).as_posix()}, config, details


def verify_checkpoint(folder, config, saved):
    import torch
    from noisynets.evaluate import load_checkpoint
    from noisynets.train import evaluate_details

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, loaded_config = load_checkpoint(folder / "model.pt")
    require(loaded_config == config, f"Конфигурация загрузчика: {folder}")
    rows = []
    for mode, expected in sorted(saved.items()):
        actual = evaluate_details(model, config["env"], 30, 86000, stochastic=mode == "resampled_noise")
        require(actual == expected, f"Повторная оценка отличается: {folder}/{mode}")
        rows.append({"condition": config["algorithm"], "seed": config["seed"], "mode": mode,
                     "n_episodes": len(actual), "mean_return": float(np.mean([row["return"] for row in actual])),
                     "matches": True})
    return rows


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/cartpole_dqn_stopping")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", action="store_true", help="Повторить все 900 итоговых эпизодов на CPU")
    args = parser.parse_args()
    args.results = args.results.resolve()
    manifest, freeze = read_json(args.results / "manifest.json"), read_json(args.results / "freeze.json")
    expected_sha = hashlib.sha256((args.results / "manifest.json").read_bytes()).hexdigest()
    require(expected_sha == freeze["manifest_sha256"], "Манифест изменился после freeze")
    require(manifest["frozen_training_source_sha256"] == freeze["training_source_sha256"], "Freeze source mismatch")
    require(manifest["seeds"] == list(range(410, 420)), "Неверные проспективные seed")
    jobs = []
    for env in manifest["environments"]:
        for condition in env["conditions"]:
            base = env["config"] | {k: v for k, v in condition.items() if k != "key"}
            for seed in manifest["seeds"]:
                folder = args.results / env["key"] / condition["key"] / f"seed_{seed}"
                require((folder / "COMPLETE").is_file(), f"Серия ещё не завершена: {folder}")
                jobs.append((folder, base | {"seed": seed}))
    require(len(jobs) == 20, "Ожидаются 20 проспективных запусков")
    rows, verification = [], []
    for folder, expected in jobs:
        row, config, details = inspect_run(folder, expected, freeze)
        rows.append(row)
        if args.verify:
            verification.extend(verify_checkpoint(folder, config, details))
            print(f"Повторена модель {row['condition']}/seed_{row['seed']}", flush=True)
    conditions = []
    for condition in ["dqn", "noisy"]:
        group = [row for row in rows if row["condition"] == condition]
        require(len(group) == 10, f"Неполный набор {condition}")
        entry = {"condition": condition, "n_training_seeds": len(group),
                 "actual_steps": summarize_values([row["actual_steps"] for row in group]),
                 "total_training_steps": sum(row["actual_steps"] for row in group),
                 "validation_stops": sum(row["stop_reason"] == "validation_threshold" for row in group),
                 "stopped_before_cap": sum(row["stopped_before_cap"] for row in group),
                 "cap_exhausted": sum(row["stop_reason"] == "budget_exhausted" for row in group),
                 "heldout_all_modes_ge475": sum(row["all_modes_ge475"] for row in group), "modes": {}}
        for mode in ["mean_weights", "resampled_noise"]:
            values = [row[mode] for row in group if row[mode] is not None]
            if values:
                entry["modes"][mode] = summarize_values(values) | {"models_ge475": sum(value >= 475 for value in values)}
        conditions.append(entry)
    total = sum(row["actual_steps"] for row in rows)
    result = {"study": "prospective_validation_stopping", "n_training_runs": len(rows),
              "requested_step_cap_per_run": 160000, "total_training_steps": total,
              "maximum_training_steps": 3200000, "steps_saved_relative_to_all_caps": 3200000-total,
              "conditions": conditions, "frozen_manifest_sha256": freeze["manifest_sha256"],
              "exact_checkpoint_verification": {"performed": args.verify, "model_modes": len(verification),
                                                 "episodes": sum(row["n_episodes"] for row in verification),
                                                 "all_match": True if args.verify else None},
              "interpretation": "Отдельный протокол с выбором времени остановки по validation; fixed-budget серия сохранена отдельно"}
    output = args.out or args.results / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", rows)
    if args.verify:
        write_csv(output / "verification.csv", verification)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
