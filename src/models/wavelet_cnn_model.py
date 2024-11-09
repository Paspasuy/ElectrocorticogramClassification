import torch

import torch.nn as nn

class WaveletCNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(WaveletCNN, self).__init__()
        self.conv1d = nn.Conv1d(6, 32, kernel_size=10)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        dim = x.shape[-2]
        coefs = x[:, :, :dim-1, :]
        signal = x[:, :, -1, :]
        
        max_coef = torch.max(coefs, dim=-2)
        concat = torch.cat((max_coef.values, signal), dim=1)
        
        conv = self.conv1d(concat)
        conv = self.relu(conv)
        conv = torch.max(conv, dim=-1)[0]        
        out = self.fc(conv)
        
        return self.sigmoid(out)
        
        