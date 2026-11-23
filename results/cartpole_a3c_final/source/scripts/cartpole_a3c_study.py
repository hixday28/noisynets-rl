"""Изолированный подбор и обучение A3C на CartPole без изменения прежних серий."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from noisynets.actorcritic import PolicyConfig, create_model, evaluate_policy, train_a3c
from noisynets.train import write_csv

ROOT = Path(__file__).resolve().parents[1]


def run(values, out):
    values = dict(values)
    anneal = bool(values.pop('anneal_lr', False))
    c = PolicyConfig(**values)
    if c.env != 'CartPole-v1' or c.family != 'a3c':
        raise ValueError('Этот скрипт предназначен только для A3C на CartPole')
    c.validate()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(c.seed)
    torch.use_deterministic_algorithms(True)
    model = create_model(c, 4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    (out / 'config.json').write_text(json.dumps(asdict(c), indent=2) + '\n')
    (out / 'treatment.json').write_text(json.dumps({'anneal_lr': anneal,
        'schedule': 'lr = initial_lr * max(0.05, 1 - step / budget); changes after evaluation barriers'
        if anneal else 'constant'}, indent=2) + '\n')
    files = ['src/noisynets/actorcritic.py', 'src/noisynets/policy_networks.py',
             'src/noisynets/networks.py', 'scripts/cartpole_a3c_study.py']
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seed': c.seed, 'device': 'cpu', 'torch_threads': 1, 'python': sys.version,
        'versions': {p: version(p) for p in ['torch', 'gymnasium', 'numpy']},
        'training_source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'async_reproducibility': 'Thread scheduling is asynchronous and not bitwise deterministic',
        'primary_evaluation': 'behaviour: categorical sampling; rollout-held noise for noisy models',
        'optimizer': 'Adam', 'anneal_lr': anneal}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    history = []
    modes = ['behaviour', 'mean_greedy', 'mean_sampled']

    def checkpoint(step):
        scores = {}
        for mode in modes:
            details = evaluate_policy(model, c, c.eval_episodes, c.eval_seed, mode)
            scores[mode] = [r['return'] for r in details]
        row = {'step': step, 'return_mean': float(np.mean(scores['behaviour'])),
               'return_std': float(np.std(scores['behaviour'], ddof=1)),
               'greedy_mean': float(np.mean(scores['mean_greedy'])),
               'mean_sampled_mean': float(np.mean(scores['mean_sampled'])),
               'learning_rate': optimizer.param_groups[0]['lr'], **model.sigma_stats()}
        history.append(row)
        write_csv(out / 'evaluations.csv', history)
        print(f"{c.algorithm} {c.generation} seed={c.seed} step={step} "
              f"behaviour={row['return_mean']:.2f} greedy={row['greedy_mean']:.2f} "
              f"mean_sampled={row['mean_sampled_mean']:.2f}", flush=True)
        if anneal:
            for group in optimizer.param_groups:
                group['lr'] = c.learning_rate * max(0.05, 1-step/c.steps)

    started = time.perf_counter()
    checkpoint(0)
    episodes, updates, extra = train_a3c(c, model, optimizer, checkpoint)
    final = {mode: evaluate_policy(model, c, c.final_episodes, c.final_seed, mode) for mode in modes}
    (out / 'final_episode_details.json').write_text(json.dumps(final, indent=2) + '\n')
    (out / 'final_returns.json').write_text(json.dumps({m: [r['return'] for r in d]
        for m, d in final.items()}, indent=2) + '\n')
    write_csv(out / 'episodes.csv', episodes)
    write_csv(out / 'updates.csv', updates)
    torch.save({'model': model.state_dict(), 'config': asdict(c), 'obs_dim': 4,
                'n_actions': 2, 'step': c.steps}, out / 'model.pt')
    metadata.update(extra)
    metadata.update({'elapsed_seconds': time.perf_counter()-started,
                     'completed_utc': datetime.now(timezone.utc).isoformat()})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'COMPLETE').write_text('success\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    values = json.loads(args.config.read_text())
    if args.seed is not None:
        values['seed'] = args.seed
    run(values, args.out)


if __name__ == '__main__':
    main()
