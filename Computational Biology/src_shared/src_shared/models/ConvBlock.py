import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.conv = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
