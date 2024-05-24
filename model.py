""" We will create a simple neural network :)
"""

import torch


class MetaModel(torch.nn.Module):
    def __init__(self, input_size=1, n_layers=3, hidden_size=64, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Activation functions
        self.relu = torch.nn.ReLU()

        # model layers
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_size)])
        for _ in range(n_layers - 2):
            self.fc_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_layers.append(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.fc_layers[:-2]:
            x = layer(x)
            x = self.relu(x)
        x = self.fc_layers[-1](x)
        return x
