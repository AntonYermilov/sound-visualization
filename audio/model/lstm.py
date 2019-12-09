import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1)


class AudioLSTMEncoder(nn.Module):
    def __init__(self, n_mfcc: int, n_hidden: int, n_out: int):
        super().__init__()

        self.n_mfcc = n_mfcc
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.transform = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            Flatten()
        )
        self.lstm = nn.LSTM(input_size=4 * self.n_mfcc, hidden_size=self.n_hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(in_features=self.n_hidden, out_features=self.n_out),
            nn.Tanh()
        )

    def forward(self, x):
        n, c, h, w = x.shape

        x = x.transpose(2, 3).transpose(1, 2)
        assert x.shape == (n, w, c, h)

        frames = x.reshape(n * w, c, h)
        frames = self.transform(frames)

        frames = frames.view(n, w, -1)
        lstm_out, _ = self.lstm(frames)

        h = self.out(lstm_out[:, -1])
        return h
