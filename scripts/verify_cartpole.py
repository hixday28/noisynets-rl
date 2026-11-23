"""Повторная оценка всех сохранённых режимов итоговых моделей CartPole на CPU.

Примеры из корня проекта:
    .venv/bin/python scripts/verify_cartpole.py --family dqn
    .venv/bin/python scripts/verify_cartpole.py --family a3c ppo --workers 2
    .venv/bin/python scripts/verify_cartpole.py --family all

Проверка читает готовые модели и первичные результаты; обучение не запускается.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "dqn": ("cartpole_dqn_final/cartpole", ("dqn", "noisy")),
    "a3c": ("cartpole_a3c_final/cartpole", ("a3c", "noisy_a3c_factorized", "noisy_a3c_independent")),
    "ppo": ("cartpole_ppo_final", ("ppo", "noisy_ppo_factorized", "noisy_ppo_independent")),
}
TRAINING_SEEDS = tuple(range(400, 410))
FINAL_SEED = 85000
FINAL_EPISODES = 30
CSV_FIELDS = ["family", "condition", "seed", "mode", "n_episodes", "mean_return",
              "saved_mean_return", "matches", "checkpoint", "error", "first_mismatch"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def discover_runs(results, families):
    jobs, pending, unexpected = [], [], []
    for family in families:
        relative, conditions = FAMILIES[family]
        base = Path(results) / relative
        expected_names = {f"seed_{seed}" for seed in TRAINING_SEEDS}
        for condition in conditions:
            folder = base / condition
            for path in folder.glob("seed_*"):
                if path.is_dir() and path.name not in expected_names:
                    unexpected.append(display_path(path))
            for seed in TRAINING_SEEDS:
                run = folder / f"seed_{seed}"
                job = {"family": family, "condition": condition, "seed": seed, "folder": str(run)}
                if (run / "COMPLETE").is_file():
                    jobs.append(job)
                else:
                    pending.append(display_path(run))
    require(not unexpected, "Неожиданные seed в итоговой серии: " + ", ".join(unexpected))
    return jobs, pending


def validate_records(records, scores, mode):
    require(isinstance(records, list) and len(records) == FINAL_EPISODES,
            f"{mode}: необходимо ровно {FINAL_EPISODES} эпизодов")
    require(isinstance(scores, list) and len(scores) == FINAL_EPISODES,
            f"{mode}: неправильное число наград")
    require([row["seed"] for row in records] == list(range(FINAL_SEED, FINAL_SEED + FINAL_EPISODES)),
            f"{mode}: неправильные seed итоговых эпизодов")
    require([row["return"] for row in records] == scores,
            f"{mode}: final_returns не совпадает с подробностями эпизодов")
    for index, row in enumerate(records):
        length, reward = row["length"], row["return"]
        require(type(length) is int and 1 <= length <= 500,
                f"{mode}/{index}: некорректная длина эпизода")
        require(isinstance(reward, (int, float)) and math.isfinite(reward) and reward == length,
                f"{mode}/{index}: награда CartPole должна совпадать с длиной")
        require(type(row["terminated"]) is bool and type(row["truncated"]) is bool,
                f"{mode}/{index}: некорректные признаки завершения")
        require(row["terminated"] or row["truncated"], f"{mode}/{index}: эпизод не завершён")
        require(row["truncated"] == (length == 500), f"{mode}/{index}: неверный TimeLimit")


def first_difference(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return json.dumps({"episode_index": index, "saved": left, "replayed": right}, ensure_ascii=False)
    if len(expected) != len(actual):
        return json.dumps({"saved_count": len(expected), "replayed_count": len(actual)})
    return ""


def load_model(family, checkpoint, saved_config):
    if family == "dqn":
        from noisynets.evaluate import load_checkpoint
    elif family == "ppo" and saved_config.get("architecture") == "separate_orthogonal":
        from noisynets.cartpole_ppo import load_checkpoint
    else:
        require("architecture" not in saved_config,
                f"Неизвестная архитектура: {saved_config.get('architecture')}")
        from noisynets.actorcritic import load_policy_checkpoint as load_checkpoint
    return load_checkpoint(checkpoint)


def error_row(job, mode="", error=""):
    return {"family": job["family"], "condition": job["condition"], "seed": job["seed"],
            "mode": mode, "n_episodes": 0, "mean_return": None, "saved_mean_return": None,
            "matches": False, "checkpoint": display_path(Path(job["folder"]) / "model.pt"),
            "error": error, "first_mismatch": ""}


def verify_run(job):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    folder, family = Path(job["folder"]), job["family"]
    try:
        config = read_json(folder / "config.json")
        expected_algorithm = ("noisy" if job["condition"] == "noisy" else "dqn") if family == "dqn" else (
            "noisy-" + family if job["condition"].startswith("noisy_") else family)
        require(config["algorithm"] == expected_algorithm, "Алгоритм не совпадает с каталогом")
        if "_factorized" in job["condition"] or "_independent" in job["condition"]:
            require(config.get("generation") == job["condition"].rsplit("_", 1)[1],
                    "Генератор шума не совпадает с каталогом")
        require(config["env"] == "CartPole-v1" and config["seed"] == job["seed"],
                "Среда или seed обучения не совпадают с каталогом")
        require(config["final_seed"] == FINAL_SEED and config["final_episodes"] == FINAL_EPISODES,
                "Итоговая оценка должна использовать seed 85000–85029")
        saved_details = read_json(folder / "final_episode_details.json")
        saved_returns = read_json(folder / "final_returns.json")
        require(isinstance(saved_details, dict) and set(saved_details) == set(saved_returns),
                "Набор режимов details/returns различается")
        noisy = expected_algorithm.startswith("noisy")
        if family == "dqn":
            required = {"mean_weights", "resampled_noise"} if noisy else {"mean_weights"}
            allowed = required
        else:
            required = {"behaviour", "mean_greedy", "mean_sampled"} if noisy else {"behaviour", "mean_greedy"}
            allowed = {"behaviour", "mean_greedy", "mean_sampled"}
        require(required <= set(saved_details) <= allowed, "Неполный или неизвестный набор режимов")
        checkpoint = folder / "model.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["config"] == config and payload["step"] == config["steps"],
                "Конфигурация/шаг checkpoint не совпадает с итоговым запуском")
        require(payload["obs_dim"] == 4 and payload["n_actions"] == 2, "Неверные размеры модели CartPole")
        require(all(torch.isfinite(tensor).all().item() for tensor in payload["model"].values()),
                "Checkpoint содержит NaN/Inf")
        model, loaded_config = load_model(family, checkpoint, config)
        loaded_values = asdict(loaded_config) if is_dataclass(loaded_config) else loaded_config
        require(loaded_values == config, "Загрузчик восстановил другую конфигурацию")
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        original_mode = model.training
        rows = []
        for mode in sorted(saved_details):
            row = error_row(job, mode)
            try:
                records, scores = saved_details[mode], saved_returns[mode]
                validate_records(records, scores, mode)
                if family == "dqn":
                    from noisynets.train import evaluate_details
                    actual = evaluate_details(model, config["env"], FINAL_EPISODES, FINAL_SEED,
                                              stochastic=mode == "resampled_noise")
                else:
                    from noisynets.actorcritic import evaluate_policy
                    actual = evaluate_policy(model, loaded_config, FINAL_EPISODES, FINAL_SEED, mode)
                actual_scores = [record["return"] for record in actual]
                validate_records(actual, actual_scores, mode)
                mismatch = first_difference(records, actual)
                matches = not mismatch
                row.update(n_episodes=len(actual), mean_return=sum(actual_scores) / len(actual_scores),
                           saved_mean_return=sum(scores) / len(scores), matches=matches,
                           error="" if matches else "Подробности повторных эпизодов отличаются",
                           first_mismatch=mismatch)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
        require(model.training == original_mode and all(torch.equal(model.state_dict()[key], tensor)
                                                       for key, tensor in before.items()),
                "Оценка изменила состояние исходной модели")
        return rows
    except Exception as error:
        return [error_row(job, error=f"{type(error).__name__}: {error}")]


def write_outputs(output, rows, summary):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", nargs="+", choices=["all", *FAMILIES], default=["all"])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="CSV; рядом сохраняется итоговая JSON-сводка")
    parser.add_argument("--workers", type=int, default=1, help="Число отдельных процессов: от 1 до 3")
    parser.add_argument("--allow-incomplete", action="store_true", help="Проверить готовые модели, явно указав незавершённые")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 3:
        parser.error("--workers должен быть от 1 до 3")
    if "all" in args.family and len(args.family) != 1:
        parser.error("all используется отдельно от названий семейств")
    families = list(FAMILIES) if args.family == ["all"] else list(dict.fromkeys(args.family))
    suffix = "" if len(families) == len(FAMILIES) else "_" + "_".join(families)
    output = args.output or args.results / "cartpole_analysis" / f"checkpoint_verification{suffix}.csv"
    jobs, pending = discover_runs(args.results, families)
    expected_models = sum(len(FAMILIES[family][1]) * len(TRAINING_SEEDS) for family in families)
    started = datetime.now(timezone.utc).isoformat()
    summary = {"started_utc": started, "families": families, "expected_models": expected_models,
               "completed_models": len(jobs), "pending": pending, "workers": args.workers,
               "torch_threads_per_process": 1, "evaluation_seed_start": FINAL_SEED,
               "episodes_per_mode": FINAL_EPISODES, "exact_episode_details": True}
    if pending and not args.allow_incomplete:
        summary.update(all_verified=False, verification_started=False, verified_models=0,
                       verified_model_modes=0, verified_episodes=0,
                       error="Серия не завершена; для проверки части используйте --allow-incomplete")
        write_outputs(output, [], summary)
        print(f"Проверка не запущена: готовы {len(jobs)}/{expected_models} моделей. Сводка: {output.with_suffix('.json')}")
        return 1
    rows = []
    if args.workers == 1:
        for job in jobs:
            rows.extend(verify_run(job))
            print(f"Проверено: {job['family']}/{job['condition']}/seed_{job['seed']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(verify_run, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as error:
                    rows.append(error_row(job, error=f"{type(error).__name__}: {error}"))
                print(f"Проверено: {job['family']}/{job['condition']}/seed_{job['seed']}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["condition"], row["seed"], row["mode"]))
    failed_models = {(row["family"], row["condition"], row["seed"]) for row in rows if not row["matches"]}
    matches = [row for row in rows if row["matches"]]
    summary.update(completed_utc=datetime.now(timezone.utc).isoformat(), verification_started=True,
                   verified_models=len(jobs) - len(failed_models), verified_model_modes=len(matches),
                   verified_episodes=sum(row["n_episodes"] for row in matches),
                   failed_models=len(failed_models), failures=[row for row in rows if not row["matches"]],
                   all_verified=not pending and not failed_models)
    write_outputs(output, rows, summary)
    print(json.dumps({key: summary[key] for key in ["expected_models", "verified_models", "verified_model_modes",
                                                   "verified_episodes", "failed_models", "all_verified"]}, ensure_ascii=False))
    print(f"Результат: {output}")
    return 0 if not failed_models and (not pending or args.allow_incomplete) else 1


if __name__ == "__main__":
    sys.exit(main())
