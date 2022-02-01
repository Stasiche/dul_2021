from tqdm import tqdm
import numpy as np

import torch.nn as nn
from torch.optim import Adam
import torch

from src.buffer import Buffer


class EBM(nn.Module):
    def __init__(self, buffer_size, lam=0.1):
        super().__init__()
        self.buffer = Buffer(buffer_size)
        self.lam = lam
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 4),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def switch_require_grad(self, flag):
        for p in self.model.parameters():
            p.requires_grad = flag

    def sample_mcmc(self, num_samples, eps_init=10, n_steps=60, buffer_ratio=0.95):
        self.model.eval()
        self.switch_require_grad(False)

        buffer_num_samples = int(num_samples * buffer_ratio)

        x = torch.vstack([
            self.buffer.sample(buffer_num_samples, limit_size=False),
            torch.FloatTensor( (num_samples - buffer_num_samples), 1, 28, 28).uniform_(-1, 1)
        ]).to(self.device)
        x.requires_grad = True

        for step in range(n_steps):
            eps = eps_init - (eps_init * step) / n_steps

            grad = torch.autograd.grad(self.model(x).sum(), x)[0].clip(-0.03, 0.03)
            x = (x +
                 (2 * eps)**0.5 * torch.randn_like(x) * 0.005 +
                 eps * grad
                 ).clip(-1, 1)

        self.buffer.push(x.detach().cpu())
        self.switch_require_grad(True)
        self.model.train()

        return x

    def __loss(self, batch):
        noised_samples = batch + (torch.randn_like(batch) * 0.005)
        mcmc_samples = self.sample_mcmc(noised_samples.shape[0]).to(self.device)

        model_energy, noised_energy = self.model(mcmc_samples), self.model(noised_samples)

        contrastive_loss = model_energy.mean() - noised_energy.mean()
        reg_loss = (model_energy ** 2 + noised_energy ** 2).mean()

        loss = contrastive_loss + self.lam * reg_loss

        return loss, (contrastive_loss, reg_loss)

    def fit(self, train_dataloader, epochs=20, lr=1e-3):
        train_losses = []

        optim = Adam(self.model.parameters(), lr=lr, betas=(0.0, 0.999))

        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc='Training...')
            for batch, labels in pbar:
                pbar.set_postfix({'epoch': epoch})

                batch = batch.to(self.device)

                loss, (contr_loss, reg_loss) = self.__loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append([contr_loss.item(), reg_loss.item()])

        return np.array(train_losses)
