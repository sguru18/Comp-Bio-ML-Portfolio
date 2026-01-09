import torch.nn as nn
from src_shared.models import ConvBlock


# 3 conv layers followed by 2 fully connected layers with 0.5 drouput in the middle
class BaselineCNN(nn.Module):

    def __init__(self, num_classes=14):
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        self.flatten = nn.Flatten()  # because Linear requires 1d inputs

        self.fc1 = nn.Linear(
            128 * 28 * 28, 128
        )  # 128 feature maps + each is 28 x 28 (maxpooled 3x), 128 as output is arbitrary
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(x):          # x = [batch_size, 1, 224, 224]
        x = self.conv1(x)    # x = [batch_size, 32, 112, 112]
        x = self.conv2(x)    # x = [batch_size, 64, 56, 56]
        x = self.conv3(x)    # x = [batch_size, 128, 28, 28]
        x = self.flatten(x)  # x = [batch_size, 128 * 28 * 28]
        x = self.fc1(x)      # x = [batch_size, 128]
        x = self.drouput(x)
        x = self.fc2(x)      # x = [batch_size, num_classes]
        x = self.sigmoid(x)  # x = [batch_size, num_classes]
