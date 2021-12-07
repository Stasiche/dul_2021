from typing import Tuple, List
from tqdm.notebook import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        # copy mask
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self,
                 inp_dim: int,
                 hidden_dims: Tuple[int],
                 ):
        super().__init__()

        self.inp_dim = inp_dim
        self.hidden_dims = hidden_dims
        self.output_dim = inp_dim

        self.model = self.__init_model()
        self.__masking()

    def __masking(self):
        masks = []
        m = {}
        m[-1] = np.arange(self.inp_dim)
        for l in range(len(self.hidden_dims)):
            m[l] = np.random.randint(m[l - 1].min(), self.inp_dim - 1, size=self.hidden_dims[l])
            masks.append(m[l - 1][:, None] <= m[l][None, :])
        masks.append(np.repeat(m[len(self.hidden_dims) - 1][:, None] < m[-1][None, :], 2, axis=1))

        layers = [l for l in self.model.modules() if isinstance(l, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    def __init_model(self):
        layers = [MaskedLinear(self.inp_dim, self.hidden_dims[0]), nn.ReLU()]
        for in_f, out_f in zip(self.hidden_dims, self.hidden_dims[1:]):
            layers += [MaskedLinear(in_f, out_f), nn.ReLU()]
        layers += [MaskedLinear(self.hidden_dims[-1], 2*self.output_dim)]

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.model(x)
        out = out.reshape(batch_size, self.inp_dim, 2)
        return out.transpose(2, 1)