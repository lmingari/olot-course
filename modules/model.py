import torch
import torch.nn as nn

##############
### Models ###
##############

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()

        layers = []
        input_dim = in_dim

        for hidden_dim in hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FourierEmbedding(nn.Module):
    def __init__(self, dim=64, scale=10.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        n_freqs = dim // 2
        self.register_buffer("freqs", torch.randn(n_freqs) * scale)

    def forward(self, x):
        # x: (N, 1) -> (N, dim)
        proj = x * self.freqs[None, :]   # (N, dim // 2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)