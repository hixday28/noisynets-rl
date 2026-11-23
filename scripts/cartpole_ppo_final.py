"""Свежие итоговые обучения PPO по зафиксированному манифесту CartPole."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from cartpole_ppo_experiments import task

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/cartpole_ppo_final.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, help="Новый каталог результатов вместо пути из манифеста")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result_root = args.out.resolve() if args.out else ROOT / manifest["results_root"]
    plans = []
    for condition in manifest["conditions"]:
        for seed in manifest["seeds"]:
            values = {**manifest["config"], **{k:v for k,v in condition.items() if k != "key"}, "seed": seed}
            folder = result_root / condition["key"] / f"seed_{seed}"
            if (folder / "COMPLETE").exists():
                saved = json.loads((folder / "config.json").read_text())
                if saved != values:
                    raise ValueError(f"Сохранённые параметры отличаются от манифеста: {folder}")
            elif folder.exists():
                raise FileExistsError(f"Незавершённый запуск уже существует: {folder}")
            else:
                plans.append((values, folder))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, values, folder, manifest["trainer_module"]) for values, folder in plans]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
