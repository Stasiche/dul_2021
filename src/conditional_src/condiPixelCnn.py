import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm
from src.conditional_src.condiBlocks import CondiMaskedConv, CondiResidualBlock


class CondiPixelCNN(nn.Module):
    def __init__(self, input_shape, num_classes, convs_per_channel=100, num_colors=2):
        super().__init__()
        h, w = input_shape
        c = 1
        self.input_shape = (h, w)
        self.image_channels = c
        self.num_colors = num_colors
        hidden_channels = c * convs_per_channel

        module_lst = [
            CondiMaskedConv(num_classes, mask_type='A',
                            in_channels=c, out_channels=hidden_channels, kernel_size=7, padding=3)
        ]
        module_lst += [CondiResidualBlock(hidden_channels, num_classes)] * 7

        module_lst += [CondiMaskedConv(num_classes, mask_type='B',
                                       in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1),
                       nn.ReLU(),
                       CondiMaskedConv(num_classes, mask_type='B',
                                       in_channels=hidden_channels, out_channels=c * num_colors, kernel_size=1)
                       ]

        self.module_lst = nn.ModuleList(module_lst)

    @property
    def device(self):
        return next(self.module_lst.parameters()).device

    def forward(self, x, classes_one_hot):
        out = x.clone()
        for i, module in enumerate(self.module_lst):
            if isinstance(module, (CondiMaskedConv, CondiResidualBlock)):
                out = module.custom_forward(out, classes_one_hot)
            elif isinstance(module, nn.ReLU):
                out = module(out)

        # just some shape magic, trust me, im doctor 💊
        return out.reshape(x.shape[0], self.image_channels, self.num_colors, *self.input_shape).permute(0, 2, 1, 3, 4)

    def predict_proba(self, x, classes_one_hot):
        with torch.no_grad():
            return F.softmax(self(x, classes_one_hot), dim=1)

    def _step(self, batch):
        x_b = batch[0].to(self.device)
        classes_one_hot_b = batch[1].to(self.device)
        return F.cross_entropy(self(x_b, classes_one_hot_b), x_b.long())

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

    def sample(self, n, classes_one_hot):
        sample = torch.zeros(n, self.image_channels, *self.input_shape).to(self.device)
        classes_one_hot = classes_one_hot.to(self.device)
        with torch.no_grad():
            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    for c in range(self.image_channels):
                        probs = self.predict_proba(sample, classes_one_hot)[..., c, i, j]
                        sample[:, c, i, j] = torch.multinomial(probs, 1).flatten()

        return sample.cpu().numpy().transpose(0, 2, 3, 1)
