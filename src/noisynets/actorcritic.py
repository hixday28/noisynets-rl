"""A3C с асинхронными actor-learner и PPO с условной шумовой политикой."""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from .policy_networks import ActorCritic, categorical_action
from .train import write_csv


@dataclass
class PolicyConfig:
    env: str = "CartPole-v1"
    algorithm: str = "ppo"
    seed: int = 200
    steps: int = 100000
    hidden: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    sigma0: float = 0.1
    generation: str = "factorized"
    workers: int = 4
    a3c_rollout: int = 20
    ppo_rollout: int = 128
    ppo_epochs: int = 4
    minibatch: int = 64
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gradient_clip: float = 0.5
    eval_interval: int = 10000
    eval_episodes: int = 10
    final_episodes: int = 30
    eval_seed: int = 55000
    final_seed: int = 65000

    @property
    def noisy(self):
        return self.algorithm.startswith("noisy-")

    @property
    def family(self):
        return self.algorithm.removeprefix("noisy-")

    def validate(self):
        if self.algorithm not in {"a3c", "noisy-a3c", "ppo", "noisy-ppo"}:
            raise ValueError("Неизвестный алгоритм")
        if self.generation not in {"factorized", "independent"}:
            raise ValueError("Неизвестный генератор шума")
        for key in ["steps", "workers", "a3c_rollout", "ppo_rollout", "ppo_epochs",
                    "minibatch", "eval_interval", "eval_episodes", "final_episodes", "hidden"]:
            if getattr(self, key) < 1:
                raise ValueError(key)
        if self.steps % self.workers or self.eval_interval % self.workers:
            raise ValueError("Бюджет и интервал оценки должны делиться на число сред")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Недопустимые gamma или lambda")
        if self.learning_rate <= 0 or self.sigma0 < 0 or self.gradient_clip <= 0:
            raise ValueError("Недопустимые параметры оптимизации")


def create_model(c, obs_dim, actions):
    return ActorCritic(obs_dim, actions, c.hidden, c.noisy, c.sigma0, c.generation)


def gae_targets(rewards, values, next_values, terminated, done, gamma, lam):
    """Bootstrap сохраняется при TimeLimit, рекурсия GAE не пересекает reset."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*(1-terminated[t])*next_values[t] - values[t]
        carry = delta + gamma*lam*(1-done[t])*carry
        advantages[t] = carry
    return advantages, advantages + values


def nstep_targets(rewards, bootstrap, gamma):
    result, current = [], bootstrap
    for reward in reversed(rewards):
        current = reward + gamma*current
        result.append(current)
    return torch.stack(list(reversed(result))).detach()


def clipped_policy_loss(logp, old_logp, advantages, clip):
    ratio = (logp-old_logp).exp()
    return -torch.minimum(ratio*advantages, ratio.clamp(1-clip, 1+clip)*advantages).mean()


@torch.no_grad()
def evaluate_policy(model, c, episodes, seed, mode="behaviour"):
    agent = deepcopy(model)
    noisy = mode == "behaviour" and c.noisy
    agent.train(noisy)
    generator = torch.Generator().manual_seed(seed + 1000000)
    env = gym.make(c.env)
    cadence = c.a3c_rollout if c.family == "a3c" else c.ppo_rollout
    records = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed+ep)
            total, length = 0.0, 0
            while True:
                if noisy and length % cadence == 0:
                    agent.reset_noise(generator)
                logits, _ = agent(torch.as_tensor(obs).unsqueeze(0))
                action = (int(logits.argmax(1).item()) if mode == "mean_greedy" else
                          int(categorical_action(logits, generator).item()))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward; length += 1
                if terminated or truncated:
                    records.append({"seed": seed+ep, "return": total, "length": length,
                                    "terminated": bool(terminated), "truncated": bool(truncated)})
                    break
    finally:
        env.close()
    return records


class AsyncWorker:
    """Самостоятельная среда, локальная сеть и генератор одного actor-learner."""
    def __init__(self, index, c, model, shared):
        self.index, self.c, self.shared = index, c, shared
        self.local = deepcopy(model)
        self.generator = torch.Generator().manual_seed(c.seed*1000+index+1700)
        self.env = gym.make(c.env)
        self.obs, _ = self.env.reset(seed=c.seed*100+index)
        self.steps, self.episode, self.total, self.length = 0, 0, 0.0, 0

    def advance(self, quota):
        c, shared = self.c, self.shared
        collected = 0
        while collected < quota:
            with shared["lock"]:
                self.local.load_state_dict(shared["model"].state_dict())
                read_version = shared["version"]
            self.local.train()
            self.local.reset_noise(self.generator)
            states, actions, rewards, completed = [], [], [], []
            for _ in range(min(c.a3c_rollout, quota-collected)):
                states.append(np.asarray(self.obs, dtype=np.float32))
                with torch.no_grad():
                    logits, _ = self.local(torch.as_tensor(self.obs).unsqueeze(0))
                    action = int(categorical_action(logits, self.generator).item())
                self.obs, reward, terminated, truncated, _ = self.env.step(action)
                actions.append(action); rewards.append(torch.tensor(float(reward)))
                self.steps += 1; collected += 1; self.length += 1; self.total += reward
                if terminated or truncated:
                    completed.append({"worker": self.index, "worker_step": self.steps,
                                      "episode": self.episode, "return": self.total,
                                      "length": self.length, "terminated": bool(terminated),
                                      "truncated": bool(truncated)})
                    self.episode += 1
                    break
            with torch.no_grad():
                _, value = self.local(torch.as_tensor(self.obs).unsqueeze(0))
                bootstrap = value.squeeze(0)*(not terminated)
            returns = nstep_targets(rewards, bootstrap, c.gamma)
            logits, values = self.local(torch.as_tensor(np.stack(states)))
            distribution = Categorical(logits=logits)
            advantage = returns-values
            policy_loss = -(distribution.log_prob(torch.tensor(actions))*advantage.detach()).mean()
            value_loss = 0.5*advantage.square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss+c.value_coef*value_loss-c.entropy_coef*entropy
            self.local.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.local.parameters(), c.gradient_clip)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise FloatingPointError("Нечисловой градиент A3C")
            # Обновление и копирование атомарны; rollout и backward выполняются параллельно.
            with shared["lock"]:
                optimizer = shared["optimizer"]; optimizer.zero_grad(set_to_none=True)
                for dest, src in zip(shared["model"].parameters(), self.local.parameters()):
                    dest.grad = src.grad.detach().clone() if src.grad is not None else None
                stale = shared["version"]-read_version
                optimizer.step(); shared["version"] += 1; shared["steps"] += len(rewards)
                shared["updates"].append({"step": shared["steps"], "update": shared["version"],
                                         "worker": self.index, "read_version": read_version,
                                         "staleness": stale, "loss": float(loss.detach()),
                                         "entropy": float(entropy.detach()), "gradient_norm": float(grad)})
                for row in completed:
                    shared["episodes"].append(row | {"global_step_at_update": shared["steps"]})
            if terminated or truncated:
                self.obs, _ = self.env.reset(); self.total, self.length = 0.0, 0


def train_a3c(c, model, optimizer, checkpoint):
    shared = {"model": model, "optimizer": optimizer, "lock": threading.Lock(),
              "version": 0, "steps": 0, "updates": [], "episodes": []}
    workers = [AsyncWorker(i, c, model, shared) for i in range(c.workers)]
    try:
        with ThreadPoolExecutor(max_workers=c.workers) as pool:
            while shared["steps"] < c.steps:
                quota = min(c.eval_interval, c.steps-shared["steps"])//c.workers
                futures = [pool.submit(worker.advance, quota) for worker in workers]
                for future in futures:
                    future.result()
                checkpoint(shared["steps"])
    finally:
        for worker in workers:
            worker.env.close()
    return shared["episodes"], shared["updates"], {
        "actor_workers": c.workers, "actor_learner_type": "asynchronous_threads",
        "gradient_updates": shared["version"],
        "max_staleness": max(r["staleness"] for r in shared["updates"]),
        "stale_gradient_updates": sum(r["staleness"]>0 for r in shared["updates"])}


def train_ppo(c, model, optimizer, checkpoint):
    envs = [gym.make(c.env) for _ in range(c.workers)]
    observations = [env.reset(seed=c.seed*100+i)[0] for i, env in enumerate(envs)]
    generator = torch.Generator().manual_seed(c.seed+2700)
    totals, lengths = np.zeros(c.workers), np.zeros(c.workers, dtype=int)
    episode_ids = np.zeros(c.workers, dtype=int)
    episodes, updates = [], []
    progress, update_id, rollouts = 0, 0, 0
    max_ratio_error = 0.0
    try:
        while progress < c.steps:
            until_eval = c.eval_interval-progress % c.eval_interval
            horizon = min(c.ppo_rollout, (c.steps-progress)//c.workers, until_eval//c.workers)
            model.train(); model.reset_noise(generator)
            states, actions, rewards, values, next_values, terminals, dones, logps = [[] for _ in range(8)]
            for _ in range(horizon):
                obs = torch.as_tensor(np.stack(observations), dtype=torch.float32)
                with torch.no_grad():
                    logits, value = model(obs)
                    action = categorical_action(logits, generator)
                    logp = Categorical(logits=logits).log_prob(action)
                next_obs, r, term, done = [], [], [], []
                for i, env in enumerate(envs):
                    nxt, reward, terminated, truncated, _ = env.step(int(action[i]))
                    next_obs.append(nxt); r.append(reward); term.append(float(terminated))
                    done.append(float(terminated or truncated))
                    totals[i] += reward; lengths[i] += 1
                    observations[i] = nxt
                    if terminated or truncated:
                        episodes.append({"worker": i, "episode": int(episode_ids[i]),
                                         "step": progress+i+1, "return": float(totals[i]),
                                         "length": int(lengths[i]), "terminated": bool(terminated),
                                         "truncated": bool(truncated)})
                        observations[i] = env.reset()[0]
                        totals[i], lengths[i] = 0, 0; episode_ids[i] += 1
                with torch.no_grad():
                    _, nxt_values = model(torch.as_tensor(np.stack(next_obs), dtype=torch.float32))
                for destination, item in zip([states, actions, rewards, values, next_values,
                                              terminals, dones, logps],
                                             [obs, action, torch.tensor(r), value, nxt_values,
                                              torch.tensor(term), torch.tensor(done), logp]):
                    destination.append(item)
                progress += c.workers
            tensors = [torch.stack(a) for a in [states, actions, rewards, values, next_values,
                                               terminals, dones, logps]]
            s, a, r, v, nv, term, done, old_logp = tensors
            advantages, returns = gae_targets(r, v, nv, term, done, c.gamma, c.gae_lambda)
            s, a, old_logp, advantages, returns = [t.flatten(0, 1) for t in [s, a, old_logp, advantages, returns]]
            advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
            # Тот же epsilon сохраняется при сборе данных, проверке ratio и всех эпохах PPO.
            with torch.no_grad():
                check_logits, _ = model(s)
                ratio_error = (Categorical(logits=check_logits).log_prob(a)-old_logp).abs().max().item()
                max_ratio_error = max(max_ratio_error, ratio_error)
                if ratio_error > 2e-5:
                    raise AssertionError("Условная вероятность PPO изменилась до оптимизации")
            losses, entropies = [], []
            for _ in range(c.ppo_epochs):
                order = torch.randperm(len(s), generator=generator)
                for start in range(0, len(s), c.minibatch):
                    idx = order[start:start+c.minibatch]
                    logits, value = model(s[idx]); distribution = Categorical(logits=logits)
                    logp = distribution.log_prob(a[idx])
                    policy_loss = clipped_policy_loss(logp, old_logp[idx], advantages[idx], c.clip)
                    value_loss = 0.5*(value-returns[idx]).square().mean()
                    entropy = distribution.entropy().mean()
                    loss = policy_loss+c.value_coef*value_loss-c.entropy_coef*entropy
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c.gradient_clip)
                    if not torch.isfinite(loss) or not torch.isfinite(grad):
                        raise FloatingPointError("Нечисловой градиент PPO")
                    optimizer.step(); update_id += 1
                    losses.append(float(loss.detach())); entropies.append(float(entropy.detach()))
            rollouts += 1
            updates.append({"step": progress, "rollout": rollouts, "gradient_updates": update_id,
                            "loss": float(np.mean(losses)), "entropy": float(np.mean(entropies)),
                            "initial_logratio_error": ratio_error, **model.sigma_stats()})
            if progress % c.eval_interval == 0 or progress == c.steps:
                checkpoint(progress)
    finally:
        for env in envs:
            env.close()
    return episodes, updates, {"gradient_updates": update_id, "rollouts": rollouts,
                                "max_initial_logratio_error": max_ratio_error,
                                "noise_reused_during_all_ppo_epochs": True}


def load_policy_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    c = PolicyConfig(**payload["config"])
    model = create_model(c, payload["obs_dim"], payload["n_actions"])
    model.load_state_dict(payload["model"])
    return model, c


def run(c, out):
    c.validate(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(c.seed)
    torch.use_deterministic_algorithms(True)
    env = gym.make(c.env)
    obs_dim, actions = int(np.prod(env.observation_space.shape)), int(env.action_space.n)
    env.close()
    model = create_model(c, obs_dim, actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    (out/'config.json').write_text(json.dumps(asdict(c), indent=2)+'\n')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {"git_commit": commit, "seed": c.seed, "device": "cpu", "torch_threads": 1,
                "python": sys.version, "versions": {p: version(p) for p in ['torch','gymnasium','numpy']},
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "training_source_sha256": {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ['actorcritic.py','policy_networks.py','networks.py']},
                "primary_evaluation": "behaviour: categorical sampling; trained noise cadence",
                "async_reproducibility": "A3C thread scheduling is not bitwise deterministic"}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    history = []
    def checkpoint(step):
        details = evaluate_policy(model, c, c.eval_episodes, c.eval_seed)
        scores = [r['return'] for r in details]
        history.append({'step':step, 'return_mean':float(np.mean(scores)),
                        'return_std':float(np.std(scores, ddof=1)), **model.sigma_stats()})
        write_csv(out/'evaluations.csv', history)
        print(f'{c.algorithm} {c.generation} seed={c.seed} step={step}: {np.mean(scores):.2f}',flush=True)
    started = time.perf_counter(); checkpoint(0)
    trainer = train_a3c if c.family == 'a3c' else train_ppo
    episodes, updates, extra = trainer(c, model, optimizer, checkpoint)
    final = {mode:evaluate_policy(model, c, c.final_episodes, c.final_seed, mode)
             for mode in (['behaviour','mean_greedy','mean_sampled'] if c.noisy else ['behaviour','mean_greedy'])}
    (out/'final_episode_details.json').write_text(json.dumps(final, indent=2)+'\n')
    (out/'final_returns.json').write_text(json.dumps({m:[r['return'] for r in d] for m,d in final.items()},indent=2)+'\n')
    write_csv(out/'episodes.csv',episodes);write_csv(out/'updates.csv',updates)
    torch.save({'model':model.state_dict(),'config':asdict(c),'obs_dim':obs_dim,
                'n_actions':actions,'step':c.steps},out/'model.pt')
    metadata.update(extra);metadata['elapsed_seconds']=time.perf_counter()-started
    metadata['completed_utc']=datetime.now(timezone.utc).isoformat()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'COMPLETE').write_text('success\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int)
    args=p.parse_args();values=json.loads(args.config.read_text())
    if args.seed is not None: values['seed']=args.seed
    run(PolicyConfig(**values),args.out)


if __name__=='__main__':
    main()
