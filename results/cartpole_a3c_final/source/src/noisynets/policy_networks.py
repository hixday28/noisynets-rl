"""Сети политики и ценности с независимым либо факторизованным шумом."""

import torch
from torch import nn

from .networks import NoisyLinear


class PolicyNoisyLinear(NoisyLinear):
    def __init__(self, inputs, outputs, sigma0=0.1, generation="factorized"):
        super().__init__(inputs, outputs, sigma0)
        if generation not in {"factorized", "independent"}:
            raise ValueError("Неизвестный способ генерации шума")
        self.generation = generation

    @torch.no_grad()
    def reset_noise(self, generator=None):
        device = self.weight_mu.device
        if self.generation == "independent":
            self.weight_epsilon.copy_(torch.randn(self.weight_epsilon.shape, device=device,
                                                  generator=generator))
            self.bias_epsilon.copy_(torch.randn(self.bias_epsilon.shape, device=device,
                                                generator=generator))
        else:
            a = torch.randn(self.in_features, device=device, generator=generator)
            b = torch.randn(self.out_features, device=device, generator=generator)
            a, b = a.sign()*a.abs().sqrt(), b.sign()*b.abs().sqrt()
            self.weight_epsilon.copy_(b.outer(a)); self.bias_epsilon.copy_(b)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, actions, hidden=64, noisy=False, sigma0=0.1,
                 generation="factorized"):
        super().__init__()
        self.noisy = noisy
        self.input = nn.Linear(obs_dim, hidden)
        def layer(i, o):
            return PolicyNoisyLinear(i, o, sigma0, generation) if noisy else nn.Linear(i, o)
        self.hidden = layer(hidden, hidden)
        self.actor = layer(hidden, actions)
        self.critic = layer(hidden, 1)

    def forward(self, obs):
        x = torch.tanh(self.input(obs))
        x = torch.tanh(self.hidden(x))
        return self.actor(x), self.critic(x).squeeze(-1)

    def reset_noise(self, generator=None):
        for layer in self.modules():
            if isinstance(layer, PolicyNoisyLinear):
                layer.reset_noise(generator)

    def sigma_stats(self):
        return {f"sigma_{kind}_{name}": float(getattr(layer, kind+"_sigma").detach().abs().mean())
                for name, layer in self.named_modules() if isinstance(layer, PolicyNoisyLinear)
                for kind in ["weight", "bias"]}


def categorical_action(logits, generator):
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
