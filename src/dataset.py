from numpy.random import randint

import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, image_dataset, crop_size=(14, 14)):
        super().__init__()

        self.image_dataset = image_dataset

        self.crop_h, self.crop_w = crop_size
        self.h, self.w = 28, 28

    def generate_mask(self):
        mask = torch.zeros((1, self.h, self.w), dtype=torch.float32)

        y = (self.h - self.crop_h) // 2
        x = (self.w - self.crop_w) // 2
        y, x = randint(y - 3, y + 3), randint(x - 3, x + 3)
        mask[:, y: y + self.crop_h, x: x + self.crop_w] = 1
        return mask

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        return self.image_dataset[index][0], self.generate_mask()
