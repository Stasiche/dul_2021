import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class CondiCustomDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))
        self.labels = one_hot(torch.tensor(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
