import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_len, hidden_len, label_len=8):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_len, hidden_size=hidden_len)
        self.layer = nn.Linear(in_features=hidden_len, out_features=label_len)

    def forward(self, X):
        # X: [seq, batch, input_size]
        # h/c: [num_layer, batch, hidden_size]
        # out: [seq, batch, hidden_size]
        out, (ht, ct) = self.lstm(X)
        out = self.layer(ht)[0]
        return out