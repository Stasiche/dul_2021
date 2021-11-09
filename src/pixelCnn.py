import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm
from src.blocks import MaskedConv, ResidualBlock


class PixelCNN(nn.Module):
    def __init__(self, input_shape, convs_per_channel=30, num_colors=4, colors_dependent=True):
        super().__init__()
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.image_channels = c
        self.num_colors = num_colors
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
                                 in_channels=hidden_channels, out_channels=c * num_colors,
                                 kernel_size=1))

        self.model = nn.Sequential(*blocks)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, x):
        out = self.model(x)
        # just some shape magic, trust me, im doctor 💊
        return out.reshape(x.shape[0], self.image_channels, self.num_colors, *self.input_shape).permute(0, 2, 1, 3, 4)

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def _step(self, batch):
        batch = batch.to(self.device)
        return F.cross_entropy(self(batch), batch.long())

    def _test(self, testloader):
        losses = []

        with torch.no_grad():
            for batch in tqdm(testloader, desc="Testing...", leave=False):
                losses.append(self._step(batch).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=20, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses["test"].append(self._test(testloader))

        for _ in trange(epochs, desc="Fitting...", leave=True):
            train_losses = []
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self._test(testloader))

        return self, losses

    def sample(self, n):
        sample = torch.zeros(n, self.image_channels, *self.input_shape).to(self.device)
        with torch.no_grad():
            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    for c in range(self.image_channels):
                        probs = self.predict_proba(sample)[..., c, i, j]
                        sample[:, c, i, j] = torch.multinomial(probs, 1).flatten()

        return sample.cpu().numpy().transpose(0, 2, 3, 1)
