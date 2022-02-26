import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from src.net2 import Net


def calc_covariance_matrix(z1, z2):
    return (z1.T @ z2) / \
           (torch.sqrt((z1 ** 2).sum(0)) * torch.sqrt((z2 ** 2).sum(0)).reshape(-1, 1))


def get_off_diag_elements(matrix):
    mask = ~torch.eye(128, dtype=torch.bool, device=matrix.device)
    return matrix.masked_select(mask)


class BTWINS(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()

        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(9),
                transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),
                transforms.RandomGrayscale(0.2),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def forward(self, x):
        return self.main(x)

    @property
    def device(self):
        return next(self.parameters()).device

    def __loss(self, batch):
        t1, t2 = self.transforms(batch), self.transforms(batch)
        z1, z2 = self.net(t1), self.net(t2)

        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        covariance_matrix = calc_covariance_matrix(z1, z2)

        invariance_loss = ((1 - covariance_matrix.diag()) ** 2).sum()
        off_diag_elements = get_off_diag_elements(covariance_matrix)

        return invariance_loss + 0.01 * (off_diag_elements ** 2).sum()

    def fit(self, trainloader, epochs=10, lr=1e-4):
        losses = []

        optim = Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            pbar = tqdm(trainloader, desc='Training...')
            for batch, _ in pbar:
                pbar.set_postfix({'epoch': epoch})
                batch = batch.to(self.device)
                loss = self.__loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.detach().cpu().numpy())

        return np.array(losses)

    @torch.no_grad()
    def encode(self, x):
        self.eval()
        return self.net(x.to(self.device))
