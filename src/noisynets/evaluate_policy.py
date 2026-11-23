"""Оценка сохраненного A3C/PPO или шумового actor-critic."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .actorcritic import evaluate_policy, load_policy_checkpoint


def load_checkpoint(path: Path):
    config = torch.load(path, map_location="cpu", weights_only=True)["config"]
    architecture = config.get("architecture")
    if architecture is None:
        return load_policy_checkpoint(path)
    if architecture == "separate_orthogonal":
        from .cartpole_ppo import load_checkpoint as load_cartpole_ppo_checkpoint
        return load_cartpole_ppo_checkpoint(path)
    raise ValueError(f"Неизвестная архитектура модели: {architecture}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--mode',choices=['behaviour','mean_sampled','mean_greedy'],default='behaviour')
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed',type=int,default=75000)
    args=p.parse_args()
    if args.episodes<1:p.error('episodes должен быть положительным')
    torch.set_num_threads(1)
    model,c=load_checkpoint(args.checkpoint)
    rows=evaluate_policy(model,c,args.episodes,args.seed,args.mode)
    print(json.dumps({'algorithm':c.algorithm,'mode':args.mode,
                      'mean_return':float(np.mean([r['return'] for r in rows])),
                      'episodes':rows},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
