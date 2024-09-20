import torch.nn as nn
import torch.nn.functional as F

from utils.macros import SAMPLE_RATE


class NetAFD(nn.Module):
    def __init__(self):
        super(NetAFD, self).__init__()
        self.channel_in = int(SAMPLE_RATE / 50) * 2
        self.channel_out = 128
        self.channels = [32, 64]
        self.dropout_rate = 0.5

        self.conv1 = nn.Conv1d(1, self.channels[0], kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(int(self.channel_in / (2 ** len(self.channels)) * self.channels[1]), self.channel_out)
        self.fc2 = nn.Linear(self.channel_out, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
