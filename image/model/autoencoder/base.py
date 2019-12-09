import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAutoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=32, num_class=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(hidden_size, self.input_size)
        self.classifier = nn.Linear(hidden_size, self.num_class)

    def forward(self, x):
        hidden_state = F.relu(self.encoder(x.flatten(start_dim=1)))
        logits = self.classifier(hidden_state)
        reconstruction = self.decode(hidden_state)
        reconstruction = reconstruction.view_as(x)
        return logits, reconstruction

    def encode(self, x):
        h = F.relu(self.encoder(x.flatten(start_dim=1)))
        return self.encoder_mu(h), self.encoder_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        return torch.sigmoid(h)

    def classify(self, z):
        return self.classifier(z)
