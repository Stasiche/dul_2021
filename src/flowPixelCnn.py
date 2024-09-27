import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm
from src.blocks import MaskedConv, ResidualBlock
from torch.distributions import Normal, Uniform
from scipy.optimize import bisect


class FlowPixelCNN(nn.Module):
    def __init__(self, input_shape, convs_per_channel=120, num_gaussians=4, colors_dependent=True):
        super().__init__()
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.image_channels = c
        self.num_gaussians = num_gaussians

        hidden_channels = c * convs_per_channel

        blocks = [MaskedConv(image_channels=c, colors_dependent=colors_dependent, mask_type='A',
                             in_channels=c, out_channels=hidden_channels,
                             kernel_size=7, padding=3)]
        for i in range(7):
            blocks.append(ResidualBlock(hidden_channels, colors_dependent))

        blocks.append(MaskedConv(image_channels=c, colors_dependent=colors_dependent,
                                 in_channels=hidden_channels, out_channels=hidden_channels,
                                 kernel_size=1))
        blocks.append(nn.ReLU())
        blocks.append(MaskedConv(image_channels=c, colors_dependent=colors_dependent,
                                 in_channels=hidden_channels, out_channels=c * 3 * num_gaussians,
                                 kernel_size=1))

        self.model = nn.Sequential(*blocks)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def base_dist(self):
        return Uniform(torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

    def forward(self, x):
        out = self.model(x)
        # just some shape magic, trust me, im doctor 💊
        return out.reshape(x.shape[0], 3 * self.num_gaussians, 1, *self.input_shape).permute(0, 2, 1, 3, 4)

    def flow(self, x):
        w_log, mu, log_s = self(x).chunk(3, dim=2)
        w = F.softmax(w_log, dim=2)
        dist = Normal(mu, log_s.exp())

        x_ = x.unsqueeze(1).repeat(1, 1, self.num_gaussians, 1, 1)
        z = (dist.cdf(x_) * w).sum(dim=2)
        log_det = (dist.log_prob(x_).exp() * w).sum(dim=2).log()

        return z, log_det

    def _loss(self, batch):
        z, log_det = self.flow(batch)
        log_prob = self.base_dist.log_prob(z).to(self.device) + log_det
        return -log_prob.mean()

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def _step(self, batch):
        batch = batch.to(self.device)
        return self._loss(batch)

    def _test(self, testloader):
        losses = []

        with torch.no_grad():
            for batch in tqdm(testloader, desc="Testing...", leave=False):
                deq_batch = batch / 2 + Uniform(0.0, 0.5).sample(batch.shape)
                losses.append(self._step(deq_batch).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=20, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses["test"].append(self._test(testloader))

        for _ in trange(epochs, desc="Fitting...", leave=True):
            train_losses = []
            for batch in trainloader:
                deq_batch = batch / 2 + Uniform(0.0, 0.5).sample(batch.shape)
                loss = self._step(deq_batch)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self._test(testloader))

        return self, losses

    def inverse(self, w, mu, log_s, z):
        dist = Normal(mu, log_s.exp())

        def f(x):
            x = torch.FloatTensor(np.repeat(x, self.num_gaussians)).to(self.device)
            return (w * dist.cdf(x)).sum() - z

        return bisect(f, -20, 20)

    @torch.no_grad()
    def sample(self):
        samples = torch.zeros(*self.input_shape, device=self.device)
        z_all = self.base_dist.sample((20, 20)).to(self.device)
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                w_log, mu, log_s = torch.chunk(self(samples[None, None, :, :]), 3, dim=2)
                w = F.softmax(w_log, dim=2)

                w = w[:, 0, :, i, j]
                mu = mu[:, 0, :, i, j]
                log_s = log_s[:, 0, :, i, j]
                z = z_all[i, j]

                samples[i, j] = self.inverse(w, mu, log_s, z)

        return samples.clip(0, 1).cpu().numpy()
