import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from tqdm.auto import trange, tqdm
from torchdiffeq import odeint_adjoint as odeint
from src.uwb_model import UWBModel
from torch import tanh, mean


class CNF(nn.Module):
    def __init__(self, inp_dim, hidden_dim, width, t1=10):
        super().__init__()
        self.width = width
        self.uwb_model = UWBModel(inp_dim, hidden_dim, width)
        self.t1 = t1
        self.t0 = 0

    @property
    def device(self):
        return next(self.uwb_model.parameters()).device

    @property
    def base_distribution(self):
        return MultivariateNormal(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))

    @staticmethod
    def get_dz_dt(z, u, w, b):
        h = tanh(z.matmul(w) + b)
        return mean(h.matmul(u), dim=0)

    def get_dlogpz_dt(self, f, z):
        """
        Честно говоря, вряд ли бы додумался, если бы не авторы статьи, забыл что производную можно считать вот так вот
        https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py#L66
        """
        trace = torch.zeros(z.shape[0], device=self.device)

        for i in range(z.shape[1]):
            trace -= torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]

        return trace.reshape(-1, 1)

    def get_uwb(self, t):
        return self.uwb_model(t)

    def forward(self, t, z):
        z = z[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.get_dz_dt(
                z.unsqueeze(0).repeat(self.width, 1, 1),
                *self.get_uwb(t)
            )

            dlogpz_dt = self.get_dlogpz_dt(dz_dt, z)

        return dz_dt, dlogpz_dt

    def flow(self, x):
        logdet_1 = torch.zeros((x.shape[0], 1)).to(self.device)

        z_steps, logdet_steps = odeint(
            self,
            (x, logdet_1),
            torch.FloatTensor([self.t1, self.t0]).to(self.device),
            atol=1e-5, rtol=1e-5, method="dopri5"
        )
        z_0, logdet_0 = z_steps[-1], logdet_steps[-1]

        return z_0, logdet_0

    def log_prob(self, batch):
        batch = batch.to(self.device)
        z, log_det = self.flow(batch)

        return self.base_distribution.log_prob(z) - log_det.flatten()

    def _step(self, batch):
        batch = batch.to(self.device)
        return -self.log_prob(batch).mean()

    def fit(self, trainloader, testloader, epochs, lr):
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        losses = {"train": [], "test": []}

        losses["test"].append(self._test(testloader))

        for _ in trange(epochs, desc="Fitting...", leave=True):
            train_losses = []
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self._test(testloader))

        return self, losses

    @torch.no_grad()
    def _test(self, testloader):
        losses = []
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            losses.append(self._step(batch).cpu().numpy())

        return np.mean(losses)

    @torch.no_grad()
    def get_probs(self, data):
        probs = []

        for batch in tqdm(data, desc="Getting probabilities...", leave=False):
            probs.append(self.log_prob(batch).exp().cpu().numpy())

        return np.hstack(probs)

    @torch.no_grad()
    def get_latents(self, data):
        latents = []

        for batch in tqdm(data, desc="Getting latents...", leave=False):
            batch = batch.to(self.device)
            latents.append(self.flow(batch)[0].cpu().numpy())

        return np.vstack(latents)
