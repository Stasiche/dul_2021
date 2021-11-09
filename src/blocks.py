import torch
from torch import nn


class MaskedConv(nn.Conv2d):
    def __init__(self, colors_dependent=True, mask_type='B', image_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))

        self.vert_center = self.kernel_size[0] // 2
        self.hor_center = self.kernel_size[1] // 2
        self.ch_per_inp = self.in_channels // image_channels
        self.ch_per_out = self.out_channels // image_channels

        if colors_dependent:
            self.set_dependent_mask(mask_type)
        else:
            self.set_independent_mask(mask_type)

    def set_independent_mask(self, mask_type):
        one_more = int(mask_type == 'B')
        self.mask[:, :, self.vert_center, :self.hor_center + one_more] = 1
        self.mask[:, :, :self.vert_center, :] = 1

    def set_dependent_mask(self, mask_type):
        self.mask[:, :, self.vert_center, :self.hor_center] = 1
        self.mask[:, :, :self.vert_center, :] = 1

        if mask_type == 'B':
            self.mask[:self.ch_per_out, :self.ch_per_inp, self.vert_center, self.hor_center] = 1
            self.mask[self.ch_per_out:2 * self.ch_per_out, :2 * self.ch_per_inp, self.vert_center, self.hor_center] = 1
            self.mask[2 * self.ch_per_out:, :, self.vert_center, self.hor_center] = 1
        else:
            self.mask[self.ch_per_out:2 * self.ch_per_out, :self.ch_per_inp, self.vert_center, self.hor_center] = 1
            self.mask[2 * self.ch_per_out:, :2 * self.ch_per_inp, self.vert_center, self.hor_center] = 1

    def forward(self, x):
        self.weight.data = self.weight.data.mul(self.mask)
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, colors_dependent=True, image_channels=3):
        super().__init__()
        out_ch = inc // 2

        self.main = nn.Sequential(
            nn.ReLU(),
            MaskedConv(colors_dependent=colors_dependent, mask_type='B',
                       image_channels=image_channels, in_channels=inc, out_channels=out_ch,
                       kernel_size=1),
            nn.ReLU(),
            MaskedConv(colors_dependent=colors_dependent, mask_type='B',
                       image_channels=image_channels, in_channels=out_ch, out_channels=out_ch,
                       kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv(colors_dependent=colors_dependent, mask_type='B',
                       image_channels=image_channels, in_channels=out_ch, out_channels=inc,
                       kernel_size=1),
        )

    def forward(self, x):
        return self.main(x) + x
