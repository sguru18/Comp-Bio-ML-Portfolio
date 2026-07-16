import torch.nn as nn

# cryptobench paper's mlp arch
# 2560 -> 256 -> 256 -> 2 via softmax
# 0.3 dropout after each of the first two layers
# L2 regularization on first two
# BCE Loss, Adam, lr = 1e-4
# batch size 2048 residues, 7 epochs

# our input embeddings are 1152 dim though instead of 2560
# going to use one logit with sigmoid isntead of 2 with softmax,
# mathematically equivalent and cleaner to handle class imbalance
# so lets do 1152 -> 256 -> 128 -> 1
# encoder goes from 1152 -> 256 -> 128, MLP takes encoder and goes 128 -> 1


class MLPEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # x is 1152 dim from esmc
        x = self.fc1(x)  # [batch_size, 1152] -> [batch_size, 256]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, 256] -> [batch_size, 128]
        x = self.relu(x)
        x = self.dropout(x)

        return x  # return the [batch_size, 128] vector for use in fusion or MLP.py
