import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from src.blocks import Encoder


class RotationPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = Encoder(latent_dim=4)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def __eval(self, trainloader):
        acc = 0
        for batch in tqdm(trainloader, desc='Evaluating...', leave=False):
            x, label = batch
            x = x.to(self.device)
            y = torch.argmax(self.net(x, None), dim=1).cpu().numpy()
            acc += accuracy_score(label.numpy(), y)

        return acc / len(trainloader)

    def fit(self, trainloader, epochs, lr):
        losses = []
        accuracy = [self.__eval(trainloader)]

        optim = Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            pbar = tqdm(trainloader, desc='Training...')
            for batch in pbar:
                pbar.set_postfix({'epoch': epoch})
                x, label = batch
                x = x.to(self.device)
                label = label.to(self.device)
                out = self.net(x, None)
                loss = self.criterion(out, label)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

            accuracy.append(self.__eval(trainloader))

        return np.array(losses), np.array(accuracy)
