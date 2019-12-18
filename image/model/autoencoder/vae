import torch
import torch.nn as nn
from enum import Enum

class EncoderState(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3

class VAE(nn.Module):
    def __init__(self, size=784, hidden_size=20, num_class=10):
        super(VAE, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.state = EncoderState.TRAIN

        self.fc1 = nn.Linear(self.size, 400)
        self.fc21 = nn.Linear(400, self.hidden_size)
        self.fc22 = nn.Linear(400, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 400)
        self.fc4 = nn.Linear(400, self.size)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_class)
        )

    def encode(self, x):
        x = x.flatten(start_dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        if self.state == EncoderState.EVAL:
            return self.decode(x)

        if self.state == EncoderState.PREDICT:
            return self.classifier(x)

        mu, logvar = self.encode(x)
        z = mu + torch.sqrt(logvar.exp()) * torch.randn(logvar.shape).to(device)
        logits = self.classifier(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        reconstruction = self.decode(mu + eps * std).view_as(x)
        return logits, reconstruction

    def __call__(self, x):
        return self.forward(x)

    def set_train(self):
        self.state = EncoderState.TRAIN

    def set_eval(self):
        self.state = EncoderState.EVAL

    def set_predict(self):
        self.state = EncoderState.PREDICT
