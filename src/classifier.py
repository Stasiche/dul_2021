from tqdm import tqdm

import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import torch


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def log_prob(self, x):
        return torch.log(torch.softmax(self.net(x), dim=1))

    def switch_require_grad(self, flag):
        for p in self.net.parameters():
            p.requires_grad = flag

    def fit(self, train_dataloader, epochs=20, lr=1e-3):
        optim = Adam(self.net.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc='Training classifier...')
            for items, labels in pbar:
                pbar.set_postfix({'epoch': epoch})
                items = items.to(self.device)
                labels = labels.to(self.device)
                out = self.net(items)
                loss = F.cross_entropy(out, labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

        return losses

