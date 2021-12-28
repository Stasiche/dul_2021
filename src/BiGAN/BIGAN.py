from torch import nn
import torch
import torch.nn.functional as F
from src.BiGAN.blocks import G, D, E
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm.auto import tqdm


class BiGAN(nn.Module):
    __lr = 2e-4
    __beta1 = 0.5
    __beta2 = 0.99
    __wd = 2e-5

    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.generator = G(x_dim, z_dim)
        self.discriminator = D(x_dim, z_dim)
        self.encoder = E(x_dim, z_dim)
        self.classifier = nn.Linear(z_dim, 10)

    def reset_cls(self):
        self.classifier = nn.Linear(self.z_dim, 10)
        self.to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def adversarial_loss(self, x_real):
        batch_size = x_real.shape[0]

        z_real = self.encoder(x_real).reshape(batch_size, -1)
        z_fake = torch.randn(batch_size, self.z_dim).type_as(x_real)

        x_real = x_real.reshape(batch_size, -1)
        x_fake = self.generator(z_fake).reshape(batch_size, -1)

        return (
                -(self.discriminator(x_real, z_real)).log().mean() - (
                1 - self.discriminator(x_fake, z_fake)).log().mean()
        )

    def fit(self, trainloader, num_epochs):
        g_optim = Adam(self.generator.parameters(), lr=self.__lr, betas=(self.__beta1, self.__beta2),
                       weight_decay=self.__wd)
        d_optim = Adam(self.discriminator.parameters(), lr=self.__lr, betas=(self.__beta1, self.__beta2),
                       weight_decay=self.__wd)
        e_optim = Adam(self.encoder.parameters(), lr=self.__lr, betas=(self.__beta1, self.__beta2),
                       weight_decay=self.__wd)

        g_scheduler = LambdaLR(g_optim, lambda epoch: (num_epochs - epoch) / num_epochs, last_epoch=-1)
        d_scheduler = LambdaLR(d_optim, lambda epoch: (num_epochs - epoch) / num_epochs, last_epoch=-1)

        losses = []
        for epoch in range(1, num_epochs + 1):

            pbar = tqdm(trainloader, desc='Training...', postfix={'epoch': epoch})
            for batch_real, _ in pbar:
                batch_real = batch_real.to(self.device)

                d_loss = self.adversarial_loss(batch_real)

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                g_loss = -self.adversarial_loss(batch_real)

                g_optim.zero_grad()
                e_optim.zero_grad()
                g_loss.backward()
                e_optim.step()
                g_optim.step()

                losses.append(d_loss.detach().cpu().numpy())
                pbar.set_postfix({'epoch': f'{epoch}/{num_epochs}', 'd_loss': d_loss.item(), 'g_loss': g_loss.item()})

            g_scheduler.step()
            d_scheduler.step()

        return np.array(losses)

    def fit_classifier(self, trainloader, num_epochs):
        classifier_optim = Adam(self.classifier.parameters(), lr=self.__lr)
        losses = []

        self.encoder.eval()

        for epoch in range(1, num_epochs + 1):
            batch_losses = []
            pbar = tqdm(trainloader, desc='Training classifier...', postfix={'epoch': epoch})
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    z = self.encoder(x)

                y_pred = self.classifier(z)
                loss = F.cross_entropy(y_pred, y)

                classifier_optim.zero_grad()
                loss.backward()
                classifier_optim.step()

                batch_losses.append(loss.detach().cpu().numpy())
                pbar.set_postfix({'epoch': f'{epoch}/{num_epochs}', 'loss': loss.item()})

            losses.append(np.mean(batch_losses))

        self.train()

        return losses

    @torch.no_grad()
    def sample(self, n):
        self.generator.eval()
        z = (torch.rand(n, self.z_dim).to(self.device) - 0.5) * 2
        self.generator.train()
        return self.generator(z).reshape(-1, 1, 28, 28).cpu()

    @torch.no_grad()
    def reconstruction(self, x):
        self.generator.eval()
        self.encoder.eval()

        z = self.encoder(x.to(self.device))
        recons = self.generator(z).reshape(-1, 1, 28, 28)

        self.train()
        return recons
