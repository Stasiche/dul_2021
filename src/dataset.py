import torch
from torch.utils.data import Dataset
from torch.distributions import Uniform


class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
