import torch
from torch import nn


class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type='B', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))

        self.vert_center = self.kernel_size[0] // 2
        self.hor_center = self.kernel_size[1] // 2

        self.set_independent_mask(mask_type)

    def set_independent_mask(self, mask_type):
        one_more = int(mask_type == 'B')
        self.mask[:, :, self.vert_center, :self.hor_center + one_more] = 1
        self.mask[:, :, :self.vert_center, :] = 1

    def forward(self, x):
        self.weight.data = self.weight.data.mul(self.mask)
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc):
        super().__init__()
        out_ch = inc // 2

        self.main = nn.Sequential(
            nn.ReLU(),
            MaskedConv(mask_type='B', in_channels=inc, out_channels=out_ch,
                       kernel_size=1),
            nn.ReLU(),
            MaskedConv(mask_type='B', in_channels=out_ch, out_channels=out_ch,
                       kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv(mask_type='B', in_channels=out_ch, out_channels=inc,
                       kernel_size=1),
        )

    def forward(self, x):
        return self.main(x) + x
