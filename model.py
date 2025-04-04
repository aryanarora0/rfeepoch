import torch
import torch as nn

class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x