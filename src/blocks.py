import torch

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU()
        )

        self.linear_block = nn.Sequential(
            nn.Linear(4 * 4 * 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = torch.flatten(out, start_dim=1)

        return self.linear_block(out)


class Encoder(nn.Module):
    def __init__(self, latent_dim=1, noise_dim=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU()
        )

        self.linear_block = nn.Linear(4 * 4 * 128 + noise_dim, latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, noise):
        out = self.conv_block(x)
        out = torch.flatten(out, start_dim=1)

        # return self.linear_block(torch.cat((out, noise), dim=1))
        return self.linear_block(out)


class Decoder(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.ReLU()
        )

        self.conv_transposed_block = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.linear_block(x)

        return self.conv_transposed_block(out.reshape(batch_size, 128, 4, 4))