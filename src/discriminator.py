import torch
from torch import nn
from src.blocks import ResnetBlockDown


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(128, 4 * 4 * 256)
        network = [
            ResnetBlockDown(3, n_filters=128),
            ResnetBlockDown(128, n_filters=128),
            ResnetBlockDown(128, n_filters=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
        ]
        self.net = nn.Sequential(*network)
        self.fc = nn.Linear(128, 1)

    @property
    def device(self):
        return next(self.net.parameters()).device

    def forward(self, z):
        z = self.net(z)
        z = torch.sum(z, dim=(2, 3))
        return self.fc(z)
