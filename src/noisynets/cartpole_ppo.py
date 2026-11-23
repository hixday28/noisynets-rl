"""PPO с раздельными actor/critic для отдельной серии настройки CartPole."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .actorcritic import PolicyConfig, evaluate_policy, train_ppo
from .policy_networks import PolicyNoisyLinear
from .train import write_csv


@dataclass
class CartPolePPOConfig(PolicyConfig):
    architecture: str = "separate_orthogonal"
    anneal_learning_rate: bool = True
    adam_epsilon: float = 1e-5

    def validate(self):
        super().validate()
        if self.family != "ppo" or self.env != "CartPole-v1":
            raise ValueError("Этот протокол предназначен для PPO на CartPole")
        if self.architecture != "separate_orthogonal" or self.adam_epsilon <= 0:
            raise ValueError("Некорректные настройки раздельной сети")


class SeparateActorCritic(nn.Module):
    def __init__(self, observations, actions, c):
        super().__init__()
        self.noisy = c.noisy

        def linear(inputs, outputs, noisy, gain):
            layer = (PolicyNoisyLinear(inputs, outputs, c.sigma0, c.generation)
                     if noisy else nn.Linear(inputs, outputs))
            weight = layer.weight_mu if noisy else layer.weight
            bias = layer.bias_mu if noisy else layer.bias
            nn.init.orthogonal_(weight, gain)
            nn.init.zeros_(bias)
            return layer

        def branch(outputs, gain):
            return nn.Sequential(linear(observations, c.hidden, False, math.sqrt(2)), nn.Tanh(),
                                 linear(c.hidden, c.hidden, c.noisy, math.sqrt(2)), nn.Tanh(),
                                 linear(c.hidden, outputs, c.noisy, gain))

        self.actor = branch(actions, .01)
        self.critic = branch(1, 1.0)

    def forward(self, observations):
        return self.actor(observations), self.critic(observations).squeeze(-1)

    def reset_noise(self, generator=None):
        for layer in self.modules():
            if isinstance(layer, PolicyNoisyLinear):
                layer.reset_noise(generator)

    def sigma_stats(self):
        return {f"sigma_{kind}_{name}": float(getattr(layer, kind+"_sigma").detach().abs().mean())
                for name, layer in self.named_modules() if isinstance(layer, PolicyNoisyLinear)
                for kind in ["weight", "bias"]}


def update_budget(c):
    progress, updates = 0, 0
    while progress < c.steps:
        horizon = min(c.ppo_rollout, (c.steps-progress)//c.workers,
                      (c.eval_interval-progress % c.eval_interval)//c.workers)
        updates += c.ppo_epochs * math.ceil(horizon*c.workers/c.minibatch)
        progress += horizon*c.workers
    return updates


class AnnealedAdam(torch.optim.Adam):
    def __init__(self, parameters, c):
        super().__init__(parameters, lr=c.learning_rate, eps=c.adam_epsilon)
        self.initial_lr, self.anneal = c.learning_rate, c.anneal_learning_rate
        self.completed_updates, self.total_updates = 0, update_budget(c)

    def step(self, closure=None):
        fraction = self.completed_updates / self.total_updates
        for group in self.param_groups:
            group["lr"] = self.initial_lr * (1-fraction) if self.anneal else self.initial_lr
        result = super().step(closure)
        self.completed_updates += 1
        return result


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    c = CartPolePPOConfig(**data["config"])
    model = SeparateActorCritic(data["obs_dim"], data["n_actions"], c)
    model.load_state_dict(data["model"])
    return model, c


def run(c, out):
    c.validate()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(c.seed)
    torch.use_deterministic_algorithms(True)
    env = gym.make(c.env)
    observations, actions = int(np.prod(env.observation_space.shape)), int(env.action_space.n)
    env.close()
    model = SeparateActorCritic(observations, actions, c)
    optimizer = AnnealedAdam(model.parameters(), c)
    (out / "config.json").write_text(json.dumps(asdict(c), indent=2)+"\n", encoding="utf-8")
    sources = ["cartpole_ppo.py", "actorcritic.py", "policy_networks.py", "networks.py", "train.py"]
    metadata = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": c.seed, "device": "cpu", "torch_threads": 1, "python": sys.version,
        "versions": {p: version(p) for p in ["torch", "gymnasium", "numpy"]},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": c.architecture,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "training_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in sources},
        "primary_evaluation": "behaviour: categorical sampling; rollout noise cadence",
        "checkpoint_selection": "last checkpoint at a fixed training budget",
        "optimizer": "Adam; linear learning-rate annealing by gradient update" if c.anneal_learning_rate else "Adam",
    }
    history = []

    def checkpoint(step):
        records = evaluate_policy(model, c, c.eval_episodes, c.eval_seed)
        scores = [r["return"] for r in records]
        history.append({"step": step, "return_mean": float(np.mean(scores)),
                        "return_std": float(np.std(scores, ddof=1)), **model.sigma_stats()})
        write_csv(out / "evaluations.csv", history)
        print(f"{c.algorithm} {c.generation} seed={c.seed} step={step}: {np.mean(scores):.2f}", flush=True)

    started = time.perf_counter()
    checkpoint(0)
    episodes, updates, extra = train_ppo(c, model, optimizer, checkpoint)
    modes = ["behaviour", "mean_greedy", "mean_sampled"] if c.noisy else ["behaviour", "mean_greedy"]
    details = {mode: evaluate_policy(model, c, c.final_episodes, c.final_seed, mode) for mode in modes}
    (out / "final_episode_details.json").write_text(json.dumps(details, indent=2)+"\n", encoding="utf-8")
    (out / "final_returns.json").write_text(json.dumps({mode: [r["return"] for r in records]
                                                       for mode, records in details.items()}, indent=2)+"\n")
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "updates.csv", updates)
    torch.save({"model": model.state_dict(), "config": asdict(c), "obs_dim": observations,
                "n_actions": actions, "step": c.steps}, out / "model.pt")
    metadata.update(extra)
    metadata.update(elapsed_seconds=time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (out / "COMPLETE").write_text("success\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(CartPolePPOConfig(**json.loads(args.config.read_text())), args.out)


if __name__ == "__main__":
    main()
