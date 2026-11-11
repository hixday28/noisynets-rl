"""Обучение DQN и NoisyNet-DQN на CPU с сохранением результатов каждого запуска."""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from .networks import QNetwork
from .replay import ReplayBuffer


@dataclass
class Config:
    env: str = "CartPole-v1"
    algorithm: str = "dqn"
    seed: int = 0
    steps: int = 30000
    hidden: int = 64
    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 50000
    learning_starts: int = 1000
    train_frequency: int = 4
    target_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15000
    sigma0: float = 0.5
    eval_interval: int = 3000
    eval_episodes: int = 5
    final_episodes: int = 20
    eval_seed: int = 10000
    final_seed: int = 20000
    gradient_clip: float = 10.0

    def validate(self):
        if self.algorithm not in {"dqn", "noisy", "noisy-fixed"}:
            raise ValueError("Unknown algorithm")
        for key in ["steps", "hidden", "batch_size", "replay_capacity", "train_frequency",
                    "target_interval", "epsilon_decay_steps", "eval_interval",
                    "eval_episodes", "final_episodes"]:
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.learning_starts < self.batch_size or self.replay_capacity < self.batch_size:
            raise ValueError("Warm-up and replay capacity must accommodate a batch")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Invalid gamma or epsilon bounds")
        if self.learning_rate <= 0 or self.sigma0 < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid learning rate, sigma0 or gradient clip")


def epsilon_at(step: int, c: Config) -> float:
    fraction = min(max(step / c.epsilon_decay_steps, 0.0), 1.0)
    return c.epsilon_start + fraction * (c.epsilon_end - c.epsilon_start)


def td_target(rewards, terminated, next_values, gamma):
    # Ограничение длины эпизода не выключает будущую ценность в TD-цели.
    return rewards + gamma * (1.0 - terminated) * next_values


@torch.no_grad()
def evaluate(model: QNetwork, env_id: str, episodes: int, seed: int,
             stochastic: bool = False) -> list[float]:
    """Оценка копии модели без изменения весов, буферов шума и генератора PyTorch."""
    agent = deepcopy(model)
    agent.train(stochastic)
    env = gym.make(env_id)
    returns = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000000)
            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                total, done = 0.0, False
                while not done:
                    if stochastic:
                        agent.reset_noise()
                    action = int(agent(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    done = terminated or truncated
                returns.append(float(total))
    finally:
        env.close()
    return returns


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(c: Config, out: Path):
    c.validate()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(c.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(c.seed + 100)
    env = gym.make(c.env)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action spaces are supported")
    obs, _ = env.reset(seed=c.seed)
    obs_dim, n_actions = int(np.prod(env.observation_space.shape)), int(env.action_space.n)
    noisy = c.algorithm != "dqn"
    online = QNetwork(obs_dim, n_actions, c.hidden, noisy, c.sigma0,
                      learn_sigma=c.algorithm != "noisy-fixed")
    target = deepcopy(online)
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=c.learning_rate)
    replay = ReplayBuffer(c.replay_capacity, obs_dim, c.seed + 200)
    config = asdict(c)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(), "device": "cpu",
                "torch_threads": 1, "git_commit": commit, "git_dirty": dirty,
                "versions": {p: version(p) for p in ["torch", "gymnasium", "numpy"]},
                "trainable_parameters": sum(p.numel() for p in online.parameters() if p.requires_grad),
                "evaluation": "greedy mean weights; separate environment; fixed validation starts",
                "final_evaluation": "held-out starts; mean weights and resampled noise for NoisyNet"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    history, episodes, updates = [], [], []
    episode_return, episode_length, episode_id, update_id = 0.0, 0, 0, 0
    started = time.perf_counter()
    def checkpoint_eval(step):
        scores = evaluate(online, c.env, c.eval_episodes, c.eval_seed)
        row = {"step": step, "return_mean": float(np.mean(scores)),
               "return_std": float(np.std(scores, ddof=1)) if len(scores)>1 else 0.0,
               "epsilon": epsilon_at(step, c) if not noisy else 0.0, **online.sigma_stats()}
        history.append(row)
        write_csv(out / "evaluations.csv", history)
        print(f"{c.algorithm} seed={c.seed} step={step}: {row['return_mean']:.1f}", flush=True)
    try:
        checkpoint_eval(0)
        for step in range(1, c.steps + 1):
            online.train()
            if noisy:
                online.reset_noise()
            if not noisy and rng.random() < epsilon_at(step - 1, c):
                action = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    action = int(online(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            episode_length += 1
            if terminated or truncated:
                episodes.append({"episode": episode_id, "step": step, "return": episode_return,
                                 "length": episode_length, "terminated": terminated,
                                 "truncated": truncated})
                episode_id += 1
                episode_return, episode_length = 0.0, 0
                obs, _ = env.reset()
            if step >= c.learning_starts and step % c.train_frequency == 0:
                states, actions, rewards, next_states, terminals = replay.sample(c.batch_size)
                online.reset_noise()
                target.train()
                target.reset_noise()  # Независимый шум основной и целевой сетей.
                with torch.no_grad():
                    targets = td_target(rewards, terminals, target(next_states).max(1).values, c.gamma)
                values = online(states).gather(1, actions[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(values, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(online.parameters(), c.gradient_clip)
                optimizer.step()
                update_id += 1
                if update_id % 100 == 0:
                    updates.append({"step": step, "update": update_id, "loss": float(loss.detach()),
                                    "gradient_norm": float(grad), **online.sigma_stats()})
            if step % c.target_interval == 0:
                target.load_state_dict(online.state_dict())
            if step % c.eval_interval == 0 or step == c.steps:
                checkpoint_eval(step)
        final_scores = evaluate(online, c.env, c.final_episodes, c.final_seed)
        final = {"mean_weights": final_scores}
        if noisy:
            final["resampled_noise"] = evaluate(online, c.env, c.final_episodes, c.final_seed,
                                                 stochastic=True)
        (out / "final_returns.json").write_text(json.dumps(final, indent=2) + "\n")
        write_csv(out / "episodes.csv", episodes)
        write_csv(out / "updates.csv", updates)
        torch.save({"model": online.state_dict(), "config": config, "obs_dim": obs_dim,
                    "n_actions": n_actions, "step": c.steps}, out / "model.pt")
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["gradient_updates"] = update_id
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (out / "COMPLETE").write_text("success\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["dqn", "noisy", "noisy-fixed"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sigma0", type=float)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    for key in ["algorithm", "seed", "steps", "sigma0"]:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    run(Config(**values), args.out)


if __name__ == "__main__":
    main()
