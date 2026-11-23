"""Отдельная настройка PPO на CartPole без изменения исходной серии."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def task(values, folder, trainer):
    config = folder.parent / (folder.name + ".json")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    log = folder.parent / (folder.name + ".log")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-m", trainer, "--config", str(config),
                                 "--out", str(folder)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Ошибка запуска: {os.path.relpath(folder, ROOT)}")
    scores = json.loads((folder / "final_returns.json").read_text())
    return {"path": os.path.relpath(folder, ROOT),
            **{mode: sum(values)/len(values) for mode, values in scores.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/cartpole_ppo_tuning.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plans = []
    for candidate in manifest["candidates"]:
        if args.candidate and candidate["key"] != args.candidate:
            continue
        for condition in manifest["conditions"]:
            for seed in manifest["training_seeds"]:
                values = {**manifest["common"], **{k:v for k,v in candidate.items() if k not in {"key", "trainer"}},
                          **{k:v for k,v in condition.items() if k != "key"}, "seed": seed}
                folder = ROOT / "results/cartpole_ppo_tuning" / candidate["key"] / condition["key"] / f"seed_{seed}"
                if (folder / "COMPLETE").exists():
                    saved = json.loads((folder / "config.json").read_text())
                    if any(saved.get(key) != value for key, value in values.items()):
                        raise ValueError(f"Сохранённые параметры отличаются от манифеста: {folder.relative_to(ROOT)}")
                elif folder.exists():
                    raise FileExistsError(f"Незавершённый запуск уже существует: {folder.relative_to(ROOT)}")
                else:
                    plans.append((values, folder, candidate.get("trainer", "noisynets.actorcritic")))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, values, folder, trainer) for values, folder, trainer in plans]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
