import torch
from torch import nn


class CondiMaskedConv(nn.Conv2d):
    def __init__(self, num_classes, mask_type='B', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))

        self.vert_center = self.kernel_size[0] // 2
        self.hor_center = self.kernel_size[1] // 2

        self.set_independent_mask(mask_type)

        self.class_embeds = nn.Linear(num_classes, self.out_channels)

    def set_independent_mask(self, mask_type):
        one_more = int(mask_type == 'B')
        self.mask[:, :, self.vert_center, :self.hor_center + one_more] = 1
        self.mask[:, :, :self.vert_center, :] = 1

    def custom_forward(self, x, classes_one_hot):
        self.weight.data = self.weight.data.mul(self.mask)
        classes_supplement = self.class_embeds(classes_one_hot.float())
        return super().forward(x) + classes_supplement[:, :, None, None]


class CondiResidualBlock(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        out_ch = in_ch // 2

        self.module_lst = nn.ModuleList([
            nn.ReLU(),
            CondiMaskedConv(num_classes, mask_type='B',
                            in_channels=in_ch, out_channels=out_ch,
                            kernel_size=1),
            nn.ReLU(),
            CondiMaskedConv(num_classes, mask_type='B',
                            in_channels=out_ch, out_channels=out_ch,
                            kernel_size=7, padding=3),
            nn.ReLU(),
            CondiMaskedConv(num_classes, mask_type='B',
                            in_channels=out_ch, out_channels=in_ch,
                            kernel_size=1)])

    def custom_forward(self, x, classes_one_hot):
        out = x.clone()
        for module in self.module_lst:
            if isinstance(module, CondiMaskedConv):
                out = module.custom_forward(out, classes_one_hot)
            elif isinstance(module, nn.ReLU):
                out = module(out)
        return out + x
