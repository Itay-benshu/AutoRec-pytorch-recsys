import torch
from torch import nn


class AutoRec(nn.Module):
    def __init__(self, n, hidden_size=512, init_std=0.03, binarize_after_first=False,
                 **kwargs):
        super().__init__()
        self.binarize_after_first = binarize_after_first
        self.first_linear = nn.Linear(n, hidden_size)

        # Customizing the weight initialization std, and setting initial biases to zero on both layers
        torch.nn.init.trunc_normal_(self.first_linear.weight, mean=0.0, std=init_std)
        torch.nn.init.zeros_(self.first_linear.bias)
        self.first_activation = nn.Sigmoid()
        self.second_linear = nn.Linear(hidden_size, n)
        torch.nn.init.trunc_normal_(self.second_linear.weight, mean=0.0, std=init_std)
        torch.nn.init.zeros_(self.second_linear.bias)
        self.second_activation = nn.Identity()

    def forward(self, x):
        return self.second_activation(
            self.second_linear(self.first_activation(self.first_linear(x)))
        )
