import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm
from src.pixelcnn.blocks import MaskedConv, ResidualBlock


class PixelCNN(nn.Module):
    def __init__(self, input_shape, hidden_channels, num_embeddings):
        super().__init__()
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.num_embeddings = num_embeddings
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(num_embeddings, c)

        blocks = [MaskedConv(mask_type='A',
                             in_channels=c, out_channels=hidden_channels,
                             kernel_size=7, padding=3)]
        for i in range(7):
            blocks.append(ResidualBlock(hidden_channels))

        blocks.append(MaskedConv(mask_type='B',
                                 in_channels=hidden_channels, out_channels=hidden_channels,
                                 kernel_size=1))
        blocks.append(nn.ReLU())
        blocks.append(MaskedConv(mask_type='B',
                                 in_channels=hidden_channels, out_channels=num_embeddings,
                                 kernel_size=1))

        self.model = nn.Sequential(*blocks)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, x):
        out = self.embedding(x).permute(0, 3, 1, 2)
        out = self.model(out)
        # just some shape magic, trust me, im doctor 💊
        return out.reshape(x.shape[0], self.num_embeddings, *self.input_shape)

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def _step(self, batch):
        batch = batch.to(self.device)
        return F.cross_entropy(self(batch), batch.long())

    def _test(self, testloader):
        losses = []

        with torch.no_grad():
            # for batch in tqdm(testloader, desc="Testing...", leave=False):
            for batch in testloader:
                losses.append(self._step(batch).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=20, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses["test"].append(self._test(testloader))

        pbar = trange(epochs, desc="Fitting...", leave=True)
        for _ in pbar:
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

            pbar.set_postfix({'train_loss': losses['train'][-1], 'test_loss': losses['test'][-1]})
            pbar.update()
        return losses

    def sample(self, n):
        sample = torch.zeros(n, *self.input_shape, dtype=torch.long).to(self.device)
        with torch.no_grad():
            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    probs = self.predict_proba(sample)[:, :, i, j]
                    sample[:, i, j] = torch.multinomial(probs.squeeze(), 1).flatten()

        return sample
