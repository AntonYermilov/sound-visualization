import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1)


class AudioCNN(nn.Module):
    def __init__(self, n_frames: int, n_mfcc: int, n_hidden: int, n_out: int):
        super().__init__()
        self.n_frames = n_frames
        self.n_mfcc = n_mfcc
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            Flatten(),
            nn.Linear(64 * (self.n_frames // 8) * (self.n_mfcc // 8), n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out),
            # add tanh ?
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
