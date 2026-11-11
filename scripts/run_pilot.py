"""Запуск серии экспериментов с проверкой уже сохраненных результатов."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--algorithms", nargs="+", choices=["dqn", "noisy", "noisy-fixed"],
                   default=["dqn", "noisy"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/pilot")
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    def launch(job):
        algorithm, seed = job
        out = args.out / algorithm / f"seed_{seed}"
        if (out / "COMPLETE").exists():
            stored = json.loads((out / "config.json").read_text())
            requested = json.loads(args.config.read_text()) | {"algorithm": algorithm, "seed": seed}
            if any(stored.get(k) != v for k, v in requested.items()):
                raise RuntimeError(f"Existing run uses a different config: {out}")
            print(f"Skip completed {out}", flush=True)
            return
        log = args.out / "logs" / f"{algorithm}_seed_{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            subprocess.run([sys.executable, "-m", "noisynets.train", "--config", str(args.config),
                            "--algorithm", algorithm, "--seed", str(seed), "--out", str(out)],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {algorithm} seed={seed}", flush=True)
    jobs = [(a, s) for s in args.seeds for a in args.algorithms]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(launch, jobs))


if __name__ == "__main__":
    main()
