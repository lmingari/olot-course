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