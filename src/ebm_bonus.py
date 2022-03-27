from tqdm import tqdm
import numpy as np

import torch.nn as nn
from torch.optim import Adam
import torch

from src.cbuffer import CBuffer


class EBMb(nn.Module):
    def __init__(self, buffer_size, lam=0.1):
        super().__init__()
        self.buffer = CBuffer(buffer_size)
        self.lam = lam
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
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
            torch.FloatTensor((num_samples - buffer_num_samples), 2).uniform_(-1, 1)
        ]).to(self.device)
        x.requires_grad = True

        for step in range(n_steps):
            eps = eps_init - (eps_init * step) / n_steps

            grad = torch.autograd.grad(self.model(x).sum(), x)[0].clip(-0.03, 0.03)
            x = (x +
                 (2 * eps) ** 0.5 * torch.randn_like(x) * 0.005 +
                 eps * grad
                 ).clip(-2.43, 3.05)

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

        return loss

    def fit(self, train_dataloader, epochs=20, lr=1e-3):
        train_losses = []

        optim = Adam(self.model.parameters(), lr=lr, betas=(0.0, 0.999))

        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc='Training ebm...')
            for batch, labels in pbar:
                pbar.set_postfix({'epoch': epoch})

                batch = batch.to(self.device)

                loss = self.__loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(loss.item())

        return np.array(train_losses)

    def sample(self, classifier, cls, num_samples, num_steps, alpha):
        self.switch_require_grad(False)
        classifier.switch_require_grad(False)
        self.eval()
        classifier.eval()

        buffer_num_samples = int(num_samples * 0.95)
        x = torch.vstack([
            self.buffer.sample(buffer_num_samples, limit_size=False),
            torch.FloatTensor((num_samples - buffer_num_samples), 2).uniform_(-1, 1)
        ]).to(self.device)
        x.requires_grad = True

        for i in range(num_steps):
            eps = 0.1 * (1 - i / num_steps)
            ebm_grad = torch.autograd.grad(self.model(x).sum(), x)[0].clip(-0.03, 0.03)
            clf_grad = torch.autograd.grad(classifier.log_prob(x)[:, cls].sum(), x)[0].clip(-0.03, 0.03)
            x = torch.clip(x +
                           (2 * eps) ** 0.5 * torch.randn_like(x) * 0.005 +
                           eps * (alpha * ebm_grad + (1 - alpha) * clf_grad),
                           -2.43, 3.05)
        return x
