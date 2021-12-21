import torch

from src.generator import Generator
from src.discriminator import Discriminator
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from tqdm.auto import trange, tqdm
import numpy as np
from utils.utils import show_samples


def convert_to_img(arr):
    return (arr.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5)


class WGANGP:
    __alpha = 2e-4
    __beta1 = 0
    __beta2 = 0.9
    __lambda = 10
    __n_critic = 5

    def __init__(self, device):
        self.generator = Generator(128).to(device)
        self.discriminator = Discriminator().to(device)
        self.num_epochs = None

    @property
    def device(self):
        return self.generator.device

    def mix(self, real, fake):
        noise = torch.rand(len(real), 1, 1, 1).to(self.device)
        noise = noise.expand_as(real)
        return noise * real + (1 - noise) * fake

    def gradient_penalty(self, real, fake):
        mixed = self.mix(real, fake)
        d_output = self.discriminator(mixed)
        gradients = torch.autograd.grad(outputs=d_output, inputs=mixed,
                                        grad_outputs=torch.ones(d_output.size()).to(self.device),
                                        create_graph=True)[0]

        gradients = gradients.reshape(len(real), -1)
        return torch.mean((torch.norm(gradients, dim=1) - 1) ** 2)

    def d_loss(self, batch):
        batch = batch.to(self.device)

        fake_data = self.generator.sample(len(batch))
        gp = self.gradient_penalty(batch, fake_data)
        return self.discriminator(fake_data).mean() - self.discriminator(batch).mean() + 10 * gp

    def g_loss(self, batch):
        fake_data = self.generator.sample(len(batch))
        return -self.discriminator(fake_data).mean()

    def fit(self, trainloader, gradient_steps):
        # num_epochs = self.__n_critic * gradient_steps // len(trainloader)
        num_epochs = gradient_steps // len(trainloader)
        # num_epochs = 500

        g_opt = Adam(self.generator.parameters(), lr=self.__alpha, betas=(self.__beta1, self.__beta2))
        g_shed = LambdaLR(g_opt, lambda epoch: (num_epochs - epoch) / num_epochs, last_epoch=-1)

        d_opt = Adam(self.discriminator.parameters(), lr=self.__alpha, betas=(self.__beta1, self.__beta2))
        d_shed = LambdaLR(g_opt, lambda epoch: (num_epochs - epoch) / num_epochs, last_epoch=-1)

        losses = {
            'train': [],
        }
        k = 0
        for epoch in range(1, num_epochs):
            self.generator.train()
            self.discriminator.train()

            pbar = tqdm(trainloader, postfix={'epoch': epoch})
            for batch in pbar:
                k += 1
                d_loss = self.d_loss(batch)

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                if not k % self.__n_critic:
                    g_loss = self.g_loss(batch)

                    g_opt.zero_grad()
                    g_loss.backward()
                    g_opt.step()

                    losses['train'].append(g_loss.item())

                    pbar.set_postfix({'epoch': f'{epoch}/{num_epochs}', 'd_loss': d_loss.item(), 'g_loss': g_loss.item()})
            g_shed.step()
            d_shed.step()

            if not epoch % (num_epochs//5):
                show_samples(convert_to_img(self.sample(100)) * 255.0, fname=f'results/q{epoch}_samples.png', title=f'CIFAR-10 generated samples')
        return np.array(losses["train"])

    @torch.no_grad()
    def sample(self, n_samples):
        self.generator.eval()
        self.discriminator.eval()
        return self.generator.sample(n_samples)
