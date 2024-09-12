import torch.nn.functional as F
import torch
import torch.nn as nn
from src.vqvae.blocks import ResidualBlock2d, VQ
from tqdm.auto import tqdm, trange
import numpy as np


class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=1.0):
        super().__init__()

        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, 2, 1),
            ResidualBlock2d(256),
            ResidualBlock2d(256),
        )

        self.vq = VQ(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            ResidualBlock2d(256),
            ResidualBlock2d(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def _step(self, batch):
        batch = batch.to(self.device)

        z = self.encoder(batch)
        e, e_grad_z, _ = self.vq(z)
        batch_recon = self.decoder(e_grad_z)

        recon_loss = F.mse_loss(batch_recon, batch)
        vq_loss = self.vq.loss(z, e)

        return recon_loss + vq_loss

    def fit(self, trainloader, testloader, epochs, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses["test"].append(self._test(testloader))

        pbar = trange(epochs, desc="Fitting...", leave=True)
        for _ in pbar:
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()

                losses["train"].append(loss.detach().cpu().numpy())

            losses["test"].append(self._test(testloader))
            pbar.set_postfix({'train_loss': losses['train'][-1], 'test_loss': losses['test'][-1]})
            pbar.update()

        losses["train"] = np.array(losses["train"])
        losses["test"] = np.array(losses["test"])

        return losses

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()

        test_losses = []
        # for batch in tqdm(testloader, desc="Testing...", leave=False):
        for batch in testloader:
            loss = self._step(batch)
            test_losses.append(loss.cpu().numpy())

        self.train()

        return np.mean(test_losses)

    def x_to_vq_embed(self, batch):
        batch = batch.to(self.device)
        z = self.encoder(batch)
        bs, _, h, w = z.shape
        return self.vq(z)[2].reshape(bs, h, w)

    @torch.no_grad()
    def vq_embed_to_x(self, batch):
        batch = batch.to(self.device)
        z = self.vq.embedding(batch)
        return self.decoder(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
