"""Оценка сохраненной модели на новых начальных состояниях среды."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .networks import QNetwork
from .train import evaluate


def load_checkpoint(path: Path) -> tuple[QNetwork, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    c = payload["config"]
    model = QNetwork(payload["obs_dim"], payload["n_actions"], c["hidden"],
                     noisy=c["algorithm"] != "dqn", sigma0=c["sigma0"],
                     learn_sigma=c["algorithm"] != "noisy-fixed")
    model.load_state_dict(payload["model"])
    return model, c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--noise", action="store_true")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    torch.set_num_threads(1)
    model, c = load_checkpoint(args.checkpoint)
    scores = evaluate(model, c["env"], args.episodes, args.seed, stochastic=args.noise)
    print(json.dumps({"returns": scores, "mean": float(np.mean(scores)),
                      "mode": "resampled_noise" if args.noise else "mean_weights"}, indent=2))


if __name__ == "__main__":
    main()
