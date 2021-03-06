import torch
import torch.nn as nn
from enum import Enum
import torch.nn.functional as F


class EncoderState(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


class VAE(nn.Module):
    def __init__(self, channels=1, height=28, width=28, hidden_size=20, num_class=10, device='cpu'):
        super(VAE, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.state = EncoderState.TRAIN
        self.device = device

        self.fc1 = nn.Linear(self.width * self.height, 400)
        self.fc21 = nn.Linear(400, self.hidden_size)
        self.fc22 = nn.Linear(400, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 400)
        self.fc4 = nn.Linear(400, self.width * self.height)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_class),
            nn.Softmax()
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
            n, _ = x.shape
            return self.decode(x).reshape(n, self.channels, self.height, self.width)

        if self.state == EncoderState.PREDICT:
            return self.classifier(x)

        mu, logvar = self.encode(x)
        var = logvar.exp()

        z = mu + torch.sqrt(var) * torch.randn(var.shape).to(self.device)

        logits = self.classifier(z)

        x = x.view(-1, 784)
        reconstruction = self.decode(z).view_as(x)

        kldiv_loss = 0.5 * (var + mu**2 - 1 - logvar).sum()
        decoder_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        vae_loss = kldiv_loss + decoder_loss

        return logits, vae_loss

    def set_train(self):
        self.state = EncoderState.TRAIN

    def set_eval(self):
        self.state = EncoderState.EVAL

    def set_predict(self):
        self.state = EncoderState.PREDICT
