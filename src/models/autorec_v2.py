import torch
from torch import nn
from src.models.autorec import AutoRec


class AutoRecV2(AutoRec):
    def __init__(self, n, hidden_size=512, input_noise_factor=0.1, latent_noise_factor=0.1, **kwargs):
        super().__init__(n, **dict(hidden_size=hidden_size, **kwargs))
        self.input_noise_factor = float(input_noise_factor)
        self.latent_noise_factor = float(latent_noise_factor)

    def forward(self, x):
        # Not adding the noise if we are evaluating
        noise = 1 if self.training else 0

        first = self.first_activation(self.first_linear(x + (torch.randn_like(x) * self.input_noise_factor * float(noise))))
        return self.second_activation(
            self.second_linear(first + (torch.randn_like(first) * self.latent_noise_factor * float(noise)))
        )
