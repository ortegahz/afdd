import torch
import torch.nn as nn


class NetAFD(nn.Module):
    def __init__(self):
        super(NetAFD, self).__init__()
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
