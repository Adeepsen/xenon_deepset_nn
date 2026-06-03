import torch
import torch.nn as nn

class PerClusterMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: shape (batch_size, 5)
        returns: shape (batch_size, 2)
        """
        return self.net(x)


x = [x, y, n_electrons_interface, drift_time_mean, drift_time_spread]
y = [p_main, p_alt]