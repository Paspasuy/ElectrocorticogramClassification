import torch

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int, num_filters: int = 16, kernel_size: int = 10, output_dim: int = 1):
        super(SimpleCNN, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1d(x)
        out = self.relu(out)
        out = torch.max(out, dim=2)[0]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out