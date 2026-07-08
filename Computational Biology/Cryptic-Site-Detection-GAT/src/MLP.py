import torch.nn as nn
from MLPEncoder import MLPEncoder

# just takes the output 128 dim from MLPEncoder and turns it into 1 logit


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = MLPEncoder()
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)

        return x
