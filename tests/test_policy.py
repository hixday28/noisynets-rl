"""Проверки шумовых actor-critic, асинхронности и условной цели PPO."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from noisynets.actorcritic import (PolicyConfig, clipped_policy_loss, create_model,
                                  evaluate_policy, gae_targets, load_policy_checkpoint,
                                  nstep_targets, run)
from noisynets.policy_networks import PolicyNoisyLinear


def test_independent_and_factorized_generators():
    gaussian = PolicyNoisyLinear(40, 30, generation="independent")
    gaussian.reset_noise(torch.Generator().manual_seed(10))
    assert abs(float(gaussian.weight_epsilon.mean())) < .1
    assert abs(float(gaussian.weight_epsilon.std())-1) < .1
    assert torch.linalg.matrix_rank(gaussian.weight_epsilon) > 1
    factor = PolicyNoisyLinear(40, 30, generation="factorized")
    factor.reset_noise(torch.Generator().manual_seed(10))
    assert torch.linalg.matrix_rank(factor.weight_epsilon) == 1
    state = torch.get_rng_state().clone()
    gaussian.reset_noise(torch.Generator().manual_seed(10))
    torch.testing.assert_close(state, torch.get_rng_state())
    gaussian(torch.randn(10, 40)).square().mean().backward()
    assert gaussian.weight_sigma.grad.abs().sum() > 0


def test_actor_critic_initial_means_match():
    torch.manual_seed(31)
    a = create_model(PolicyConfig(), 4, 2)
    torch.manual_seed(31)
    b = create_model(PolicyConfig(algorithm="noisy-ppo"), 4, 2)
    a.eval(); b.eval()
    obs = torch.randn(15, 4)
    for x, y in zip(a(obs), b(obs)):
        torch.testing.assert_close(x, y)


def test_nstep_and_gae_time_limits():
    result = nstep_targets([torch.tensor(1.), torch.tensor(2.)], torch.tensor(3.), .9)
    torch.testing.assert_close(result, torch.tensor([5.23, 4.7]))
    r, v, nv = torch.tensor([[1.], [2.]]), torch.tensor([[3.], [4.]]), torch.tensor([[10.], [5.]])
    adv, ret = gae_targets(r, v, nv, torch.tensor([[0.], [1.]]), torch.ones(2, 1), .9, .95)
    torch.testing.assert_close(adv, torch.tensor([[7.], [-2.]]))
    torch.testing.assert_close(ret, torch.tensor([[10.], [2.]]))


def test_ppo_ratio_uses_same_noise_and_clipping():
    c = PolicyConfig(algorithm="noisy-ppo")
    model = create_model(c, 4, 2); model.train()
    g = torch.Generator().manual_seed(20); model.reset_noise(g)
    s, actions = torch.randn(18, 4), torch.zeros(18, dtype=torch.long)
    logits, _ = model(s); old = Categorical(logits=logits).log_prob(actions).detach()
    again, _ = model(s)
    torch.testing.assert_close((Categorical(logits=again).log_prob(actions)-old).exp(), torch.ones(18))
    model.reset_noise(g); changed, _ = model(s)
    assert not torch.allclose(Categorical(logits=changed).log_prob(actions), old)
    loss = clipped_policy_loss(torch.tensor([2., .5]).log(), torch.zeros(2),
                               torch.tensor([1., -1.]), .2)
    assert float(loss) == pytest.approx(-.2)


def test_policy_evaluation_preserves_all_state():
    c = PolicyConfig(algorithm="noisy-ppo")
    model = create_model(c, 4, 2); model.train(); model.reset_noise()
    state, rng = deepcopy(model.state_dict()), torch.get_rng_state().clone()
    for mode in ['behaviour', 'mean_greedy', 'mean_sampled']:
        first = evaluate_policy(model, c, 2, 73000, mode)
        assert first == evaluate_policy(model, c, 2, 73000, mode)
        assert all(r['return']==r['length'] and 1<=r['length']<=500 for r in first)
    torch.testing.assert_close(rng, torch.get_rng_state())
    for k, value in state.items():
        torch.testing.assert_close(value, model.state_dict()[k])
    assert model.training


@pytest.mark.parametrize('algorithm', ['a3c', 'noisy-a3c', 'ppo', 'noisy-ppo'])
def test_policy_training_and_saved_evaluation(tmp_path, algorithm):
    c = PolicyConfig(algorithm=algorithm, steps=400, eval_interval=200,
                     eval_episodes=2, final_episodes=2, ppo_rollout=16,
                     hidden=16, workers=4)
    out = tmp_path/algorithm
    run(c, out)
    assert (out/'COMPLETE').exists()
    model, stored = load_policy_checkpoint(out/'model.pt')
    assert stored==c
    details=json.loads((out/'final_episode_details.json').read_text())
    for mode in details:
        assert evaluate_policy(model,c,2,c.final_seed,mode)==details[mode]
    meta=json.loads((out/'metadata.json').read_text())
    if c.family=='a3c':
        assert meta['actor_workers']==4 and meta['max_staleness']>0
    else:
        assert meta['max_initial_logratio_error']<2e-5
    assert all(meta['training_source_sha256'].values())
