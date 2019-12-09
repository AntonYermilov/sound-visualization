import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

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
        z = self.encoder(x)

        logits = self.classifier(z)
        reconstruction = self.decode(z)

        return logits, reconstruction

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def classify(self, z):
        return self.classifier(z)