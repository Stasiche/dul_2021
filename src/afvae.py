import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm, trange
from torch.distributions import Normal
from src.made import MADE


class AFVAE(nn.Module):
    def __init__(self, latent_dim=16, beta=1):
        super(AFVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.made = MADE(latent_dim, [512, 512])

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 2 * latent_dim, kernel_size=1, stride=1, padding=0)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        )

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def loss(self, batch):
        batch = batch.to(self.device)

        mu, log_sigma = torch.chunk(self.encoder(batch), 2, dim=1)
        z = Normal(0, 1).sample(mu.shape).to(self.device) * log_sigma.exp() + mu
        rec_batch = self.decoder(z)

        rec_loss = F.mse_loss(rec_batch, batch.detach(), reduction='none')
        rec_loss = rec_loss.view(len(batch), -1).sum(dim=1).mean()
        enc_log_prob = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * (z - mu) ** 2 * torch.exp(-2 * log_sigma)

        z = z.squeeze()
        mu, log_sigma = torch.chunk(self.made(z), 2, dim=1)
        mu, log_sigma = mu.squeeze(), log_sigma.squeeze()

        eps = mu + z * log_sigma.exp()
        prior_log_prob = Normal(0, 1).log_prob(eps) + log_sigma

        kl_loss = (enc_log_prob - prior_log_prob)
        kl_loss = kl_loss.sum(dim=1).mean()

        loss = rec_loss + self.beta * kl_loss

        return loss, rec_loss, kl_loss

    @torch.no_grad()
    def test(self, test_dataloader):
        total_losses, rec_losses, kl_losses = [], [], []
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            total_loss, rec_loss, kl_loss = self.loss(batch)

            total_losses.append(total_loss.item())
            rec_losses.append(rec_loss.item())
            kl_losses.append(kl_loss.item())

        return np.mean(total_losses), np.mean(rec_losses), np.mean(kl_losses)

    def fit(self, trainloader, testloader, lr=1e-3, num_epochs=100):

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        losses = {
            'test': [self.test(testloader)],
            'train': [],
        }

        for epoch in trange(num_epochs, desc='Training...'):
            for batch in trainloader:
                loss, rec_loss, kl_loss = self.loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses["train"].append([loss.item(), rec_loss.item(), kl_loss.item()])
            losses["test"].append(self.test(testloader))

        return np.array(losses["train"]), np.array(losses["test"])

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim).cuda()
        for i in range(self.latent_dim):
            mu, log_std = self.made(z)[:, :, i].chunk(2, dim=1)
            mu, log_std = mu.squeeze(-1), log_std.squeeze(-1)
            z[:, i] = (z[:, i] - mu) * torch.exp(-log_std)
        samples = self.decoder(z[:, :, None, None])
        return torch.clip(samples, -1, 1)
