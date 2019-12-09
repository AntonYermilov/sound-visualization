import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from enum import Enum

class EncoderState(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3
   
class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=1)

class FlattenDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
    def forward(self, x: torch.Tensor):
        return x.view(-1, self.hidden_size, 7, 7)

class ConvAutoencoder(nn.Module):
    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.state = EncoderState.TRAIN

        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_size // 2, self.hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.07),
            Flatten(),
            nn.Linear(self.hidden_size * 7 * 7, self.hidden_size),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 7 * 7),
            FlattenDecoder(self.hidden_size),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_size // 2, 1, kernel_size=4, stride=2, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_class)
        )

    def forward(self, x):
        # x is image
        if self.state == EncoderState.TRAIN:
            z = self.encoder(x)

            logits = self.classifier(z)
            reconstruction = torch.sigmoid(self.decoder(z))

            return logits, reconstruction
        # x is image code
        elif self.state == EncoderState.EVAL:
            return torch.sigmoid(self.decoder(x))
        # x is image code
        else:
            self.classifier(x)

    def __call__(self, x):
        return self.forward(x)

    def set_train(self):
        self.state = EncoderState.TRAIN

    def set_eval(self):
        self.state = EncoderState.EVAL

    def set_predict(self):
        self.state = EncoderState.PREDICT
