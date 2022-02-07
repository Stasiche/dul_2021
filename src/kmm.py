import torch
import torch.nn as nn

import torch.utils.data as data
import torch.optim as opt

from tqdm.auto import tqdm


class KernelMeanMatchingDRE(nn.Module):
    def __init__(self, hd=128, sigma_sq=0.1):
        super().__init__()
        self.r = nn.Sequential(nn.Linear(1, hd),
                               nn.ReLU(),
                               nn.Linear(hd, hd),
                               nn.ReLU(),
                               nn.Linear(hd, 1),
                               nn.Softplus())

        self.sigma_sq = sigma_sq

    def fit(self, data_nu, data_de, batch_size=512, lr=1e-3, num_epochs=1000):

        loader_nu = data.DataLoader(data_nu,
                                    batch_size=batch_size,
                                    shuffle=True)

        loader_de = data.DataLoader(data_de,
                                    batch_size=batch_size,
                                    shuffle=True)

        optim = opt.Adam(self.r.parameters(), lr=lr)

        for epoch in tqdm(range(num_epochs)):
            for (batch_nu, batch_de) in zip(loader_nu, loader_de):
                batch_nu = batch_nu.view(-1, 1).float()
                batch_de = batch_de.view(-1, 1).float()
                n_de, n_nu = len(batch_de), len(batch_nu)

                r_de = self.r(batch_de)

                k_dede = self.__kernel(batch_de, batch_de)
                k_denu = self.__kernel(batch_de, batch_nu)

                loss = 1 / n_de ** 2 * r_de.T @ k_dede @ r_de - \
                       2 / (n_nu * n_de) * r_de.T @ k_denu @ torch.ones(n_nu)

                optim.zero_grad()
                loss.backward()
                optim.step()

    def __kernel(self, x, y):
        return torch.exp(-torch.pow(x[:, None] - y[None, :], 2) / (2 * self.sigma_sq)).squeeze()
