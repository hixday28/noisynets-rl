"""Шумовой линейный слой по методу Fortunato и соавторов (2018)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """y = (mu_w + sigma_w * eps_w)x + mu_b + sigma_b * eps_b.

    Для всего minibatch используется один шум до вызова reset_noise().
    В режиме eval() используются средние веса; sigma обучается без ограничений.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5,
                 learn_sigma: bool = True):
        super().__init__()
        if in_features < 1 or out_features < 1 or sigma0 < 0:
            raise ValueError("Positive layer dimensions and nonnegative sigma0 required")
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma0 / math.sqrt(in_features)),
            requires_grad=learn_sigma,
        )
        # Масштаб шума смещений зависит от числа выходов слоя.
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma0 / math.sqrt(out_features)),
            requires_grad=learn_sigma,
        )
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self) -> None:
        eps_in = self.scaled_noise(self.in_features, self.weight_mu.device)
        eps_out = self.scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)


class QNetwork(nn.Module):
    """Q-сеть с двумя скрытыми слоями и шумом в двух последних линейных слоях."""

    def __init__(self, obs_dim: int, actions: int, hidden: int = 64,
                 noisy: bool = False, sigma0: float = 0.5, learn_sigma: bool = True):
        super().__init__()
        def layer(i: int, o: int) -> nn.Module:
            return NoisyLinear(i, o, sigma0, learn_sigma) if noisy else nn.Linear(i, o)
        self.layers = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                    layer(hidden, hidden), nn.ReLU(), layer(hidden, actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def reset_noise(self) -> None:
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def sigma_stats(self) -> dict[str, float]:
        return {f"sigma_abs_{name}": float(layer.weight_sigma.detach().abs().mean())
                for name, layer in self.named_modules() if isinstance(layer, NoisyLinear)}
