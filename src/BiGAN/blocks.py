from torch import nn
import torch


class G(nn.Module):
    def __init__(self, out_dim, z_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z).reshape(-1, 1, 28, 28)


class D(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim + z_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, z):
        return self.main(torch.cat((x, z), dim=1))


class E(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    def forward(self, x):
        return self.main(x.reshape(x.shape[0], -1))
