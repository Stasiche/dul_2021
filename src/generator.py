import torch
from torch import nn
from src.blocks import ResnetBlockUp
from torch.distributions import Normal


class Generator(nn.Module):
    def __init__(self, n_filters):
        super(Generator, self).__init__()
        self.fc = nn.Linear(128, 4*4*256)
        network = [
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*network)
        self.noise = Normal(torch.tensor(0.), torch.tensor(1.))

    @property
    def device(self):
        return next(self.net.parameters()).device

    def forward(self, z):
        z = self.fc(z).reshape(-1, 256, 4, 4)
        return self.net(z)

    def sample(self, n_samples):
        z = self.noise.sample([n_samples, 128]).to(self.device)
        return self.forward(z)
