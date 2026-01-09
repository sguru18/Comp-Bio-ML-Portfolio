import torch.nn as nn


# default convolution block with defaults: 3x3 kernels, 1 padding, and 2x2 windows for max pooling
# this was my first time using torch.nn (a monumental occasion)
class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3, padding=1, max_pool_size=2):
        super().__init__()

        self.conv = nn.Conv2d(channels_in, channels_out, kernel, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(max_pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
