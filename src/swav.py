import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms

from tqdm import tqdm
from src.swav_blocks import FeatureExtractor, Classifier


class SwAV(nn.Module):
    def __init__(self, features_dim, projection_dim, n_prototypes):
        super().__init__()
        self.fe = FeatureExtractor(features_dim)
        self.clf = Classifier(features_dim, projection_dim)
        self.prototypes_bank = nn.Parameter(torch.rand(projection_dim, n_prototypes))
        self.prototypes_bank.requires_grad = True

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=32),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    # https://arxiv.org/pdf/2006.09882v5.pdf p.14
    def renormalization(self, mat):
        K, B = mat.shape
        mat /= torch.sum(mat)

        r = torch.ones(K, device=self.device) / K
        c = torch.ones(B, device=self.device) / B

        for _ in range(3):
            mat *= (r / torch.sum(mat, dim=1)).unsqueeze(1)
            mat *= (c / torch.sum(mat, dim=0)).unsqueeze(0)

        mat = (mat / torch.sum(mat, dim=0, keepdim=True)).T
        return mat

    def __loss(self, batch):
        x_t = self.transforms(batch)
        x_s = self.transforms(batch)

        z = torch.cat([x_t, x_s], dim=0)
        z = self.clf(self.fe(z))
        scores = torch.mm(z, self.prototypes_bank)
        scores_t = scores[:len(batch)]
        scores_s = scores[len(batch):]

        with torch.no_grad():
            q_t = self.renormalization(torch.exp(scores_t / 0.05).T)
            q_s = self.renormalization(torch.exp(scores_s / 0.05).T)

        p_t = F.softmax(scores_t / 0.1)
        p_s = F.softmax(scores_s / 0.1)

        return -0.5 * torch.mean(q_t * torch.log(p_s) + q_s * torch.log(p_t))

    def fit(self, trainloader, epochs, lr):
        optim = Adam(self.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            pbar = tqdm(trainloader, desc='Training...')
            for batch, _ in pbar:
                pbar.set_postfix({'epoch': epoch})
                batch = batch.to(self.device)
                loss = self.__loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                with torch.no_grad():
                    self.prototypes_bank.data = F.normalize(self.prototypes_bank.data, dim=0, p=2)

                losses.append(loss.item())

        return losses

    def get_latents(self, batch):
        self.eval()
        with torch.no_grad():
            out = self.fe(batch.to(self.device))
        return out
