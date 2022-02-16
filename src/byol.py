import torch
from torch import nn
from torchvision import transforms
from copy import deepcopy
from src.net import Net

from torch.optim import Adam
from tqdm.auto import tqdm
import numpy as np
from torch.nn import functional as F


@torch.no_grad()
def exp_smoothing(src, dst, alpha):
    for src_par, dst_par in zip(src.parameters(), dst.parameters()):
        dst_par.data.copy_(alpha * dst_par.data + (1 - alpha) * src_par.data)


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = Net()
        self.student = deepcopy(self.teacher)
        self.student.requires_grad_(False)

        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(9),
            transforms.Normalize(0.5, 0.5),
        ])

        self.prediction = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def __loss(self, x, y):
        z1 = F.normalize(self.prediction(self.student(x)), dim=1)
        z2 = F.normalize(self.teacher(y), dim=1)

        return F.mse_loss(z1, z2)

    def fit_batch(self, batch):
        st_batch = self.transforms(batch)
        tr_batch = self.transforms(batch)

        return self.__loss(st_batch, tr_batch) + self.__loss(tr_batch, st_batch)

    def fit(self, trainloader, epochs=20, lr=1e-4):
        losses = []
        optim = Adam(list(self.student.parameters()) + list(self.prediction.parameters()), lr=lr)

        for epoch in range(epochs):
            pbar = tqdm(trainloader, desc='Training...')
            for batch, _ in pbar:
                batch = batch.to(self.device)
                loss = self.fit_batch(batch)
                pbar.set_postfix({'epoch': epoch, 'loss': loss.item()})

                optim.zero_grad()
                loss.backward()
                optim.step()

                exp_smoothing(self.student, self.teacher, 0.99)

                losses.append(loss.detach().cpu().numpy())

        return np.array(losses)

    @torch.no_grad()
    def encode(self, x):
        self.student.eval()
        x = transforms.Resize(24)(x).to(self.device)
        return self.student(x)
