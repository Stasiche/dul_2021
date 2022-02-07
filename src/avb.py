from tqdm.auto import trange

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from src.blocks import Classifier, Encoder, Decoder


def convert_to_image(arr):
    return (1 * (0.5 * arr.clip(-1, 1) + 0.5)).cpu().numpy()


class AVB(nn.Module):
    def __init__(self, latent_dim=32, noise_dim=32):
        super().__init__()

        self.T = Classifier(latent_dim=latent_dim)
        self.E = Encoder(latent_dim=latent_dim, noise_dim=noise_dim)
        self.D = Decoder(latent_dim=latent_dim)

        self.latent_dist = MultivariateNormal(
            torch.zeros(latent_dim),
            torch.eye(latent_dim)
        )

        self.noise_dist = MultivariateNormal(
            torch.zeros(noise_dim),
            torch.eye(noise_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def __loss(self, batch):
        z = self.E(batch, self.noise_dist.sample((len(batch),)).to(self.device))
        z_fake = self.latent_dist.sample((len(batch),)).to(self.device)
        batch_recon = self.D(z)

        rec_loss = F.mse_loss(batch, batch_recon, reduction='none')
        rec_loss = rec_loss.reshape(len(batch), -1).sum(dim=1)

        real_labels = torch.ones((len(batch), 1)).to(self.device)
        fake_labels = torch.zeros((len(batch), 1)).to(self.device)

        elbo_loss = torch.mean(rec_loss + self.T(batch, z))

        classifier_loss = torch.mean(F.binary_cross_entropy(torch.sigmoid(self.T(batch, z.detach())), real_labels) +
                                     F.binary_cross_entropy(torch.sigmoid(self.T(batch, z_fake.detach())), fake_labels))

        return elbo_loss, classifier_loss

    def fit(self, train_dataloader, test_dataloader, epochs, lr):

        train_losses, test_losses = [], []

        opt = torch.optim.Adam(
            list(self.D.parameters()) + list(self.E.parameters()), lr=lr
        )
        clf_optim = torch.optim.Adam(self.T.parameters(), lr=lr)

        test_losses.append(self._test(test_dataloader))

        for _ in trange(epochs, desc='Training...'):
            for batch in train_dataloader:
                batch = batch.to(self.device)

                elbo_loss, classifier_loss = self.__loss(batch)

                opt.zero_grad()
                elbo_loss.backward()
                opt.step()

                clf_optim.zero_grad()
                classifier_loss.backward()
                clf_optim.step()

                train_losses.append([elbo_loss.item(), classifier_loss.item()])
            test_losses.append(self._test(test_dataloader))

        return np.array(train_losses), np.array(test_losses)

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        z_sampled = self.latent_dist.sample((n,)).to(self.device)
        samples = self.D(z_sampled)
        return convert_to_image(samples)

    @torch.no_grad()
    def _test(self, test_dataloader):
        self.eval()
        total_elbo_loss, total_classifier_loss = 0, 0
        total_elements_n = 0
        for batch in test_dataloader:
            batch_size = len(batch)
            batch = batch.to(self.device)

            elbo_loss, classifier_loss = self.__loss(batch)
            total_elbo_loss += batch_size * elbo_loss.item()
            total_classifier_loss += batch_size * classifier_loss.item()
            total_elements_n += batch_size
        self.train()
        return [total_elbo_loss / total_elements_n,
                total_classifier_loss / total_elements_n]
