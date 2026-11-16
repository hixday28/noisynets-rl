"""Запуск расширенной серии по зафиксированному протоколу."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "configs/final_study.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/final")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    config_dir = args.out / "configs"
    config_dir.mkdir(exist_ok=True)
    jobs = []
    for environment in manifest["environments"]:
        for condition in environment["conditions"]:
            values = environment["config"] | {k:v for k,v in condition.items() if k != "key"}
            config = config_dir / f"{environment['key']}_{condition['key']}.json"
            content = json.dumps(values, ensure_ascii=False, indent=2) + "\n"
            if config.exists() and config.read_text() != content:
                raise RuntimeError(f"Настройки уже сохранённой серии отличаются: {config}")
            config.write_text(content)
            for seed in manifest["seeds"]:
                out = args.out / environment["key"] / condition["key"] / f"seed_{seed}"
                jobs.append((config, values | {"seed":seed}, out))
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")

    def launch(job):
        config, requested, out = job
        if (out / "COMPLETE").exists():
            stored = json.loads((out / "config.json").read_text())
            if any(stored.get(k) != v for k,v in requested.items()):
                raise RuntimeError(f"Настройки запуска отличаются: {out}")
            print(f"Пропущен готовый запуск {out.relative_to(args.out)}", flush=True)
            return
        logs = args.out / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / ("_".join(out.relative_to(args.out).parts)+".log")
        with log.open("w") as f:
            subprocess.run([sys.executable,"-m","noisynets.train","--config",str(config),
                            "--seed",str(requested["seed"]),"--out",str(out)],cwd=ROOT,
                           stdout=f,stderr=subprocess.STDOUT,check=True)
        print(f"Готово: {out.relative_to(args.out)}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(launch,jobs))


if __name__ == "__main__":
    main()
