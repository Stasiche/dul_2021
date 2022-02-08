from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from src.blocks import Discriminator, Encoder, Decoder


class ContextEncoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128):
        super().__init__()

        self.discr = Discriminator(hidden_dim=hidden_dim)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch, mask):
        z = self.encoder((1 - mask) * batch, 0)
        return self.decoder(z)

    def __recon_loss(self, batch, mask):
        batch_recon = self(batch, mask)
        recon_loss = mask * F.mse_loss(batch, batch_recon)
        recon_loss = recon_loss.reshape(batch.shape[0], -1).mean(dim=1)

        discr_fake = self.discr(batch_recon)
        fake_loss = F.binary_cross_entropy(discr_fake, torch.ones_like(discr_fake))

        loss = recon_loss + fake_loss
        return recon_loss.mean(), loss.mean()

    def __adversarial_loss(self, batch, mask):
        batch_recon = self(batch, mask).detach()

        discr_real = self.discr(batch)
        discr_fake = self.discr(batch_recon)

        real_loss = F.binary_cross_entropy(discr_real, torch.ones_like(discr_real))
        fake_loss = F.binary_cross_entropy(discr_fake, torch.zeros_like(discr_fake))
        adversarial_loss = real_loss + fake_loss

        return adversarial_loss.mean()

    def fit(self, train_dataloader, epochs, lr):
        train_losses = []
        test_losses = []

        encoder_decoder_optim = opt.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        discr_optim = opt.Adam(self.discr.parameters(), lr=lr)

        step = 1
        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc='Training...')
            for batch, mask in pbar:
                pbar.set_postfix({'epoch': epoch})

                batch = batch.to(self.device)
                mask = mask.to(self.device)
                recon_loss, loss = self.__recon_loss(batch, mask)
                adversarial_loss = self.__adversarial_loss(batch, mask)

                encoder_decoder_optim.zero_grad()
                loss.backward()
                encoder_decoder_optim.step()

                train_losses.append([recon_loss.item(), adversarial_loss.item()])
                step += 1

                if not step % 10:
                    discr_optim.zero_grad()
                    adversarial_loss.backward()
                    discr_optim.step()

        return np.array(train_losses), np.array(test_losses)

    @torch.no_grad()
    def inpaint(self, batch, mask):
        batch = batch.to(self.device)
        mask = mask.to(self.device)

        batch_recon = self(batch, mask)

        return np.vstack((
            ((1 - mask) * batch).cpu().numpy(),
            batch_recon.cpu().numpy(),
            batch.cpu().numpy()
        ))
