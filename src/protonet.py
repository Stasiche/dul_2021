import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils.hw14_utils import *


def pairwise_dist(x, y):
    return torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * x @ y.T


class ProtoNet(nn.Module):
    def __init__(self, embeddings_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, embeddings_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def get_clusters(self, imgs, labels):
        classes = torch.unique(labels)
        embeddings = self.net(imgs)

        clusters = torch.zeros(len(classes), embeddings.shape[1], device=self.device)
        for i, cls in enumerate(classes):
            clusters[i] = torch.mean(embeddings[labels == cls], dim=0)

        return clusters, classes

    def __loss(self, batch):
        train_imgs, test_imgs, train_labels, test_labels = split_batch(*batch)

        centers, classes = self.get_clusters(train_imgs, train_labels)

        dist = pairwise_dist(self.net(test_imgs), centers)

        _, test_clsses = torch.unique(test_labels, return_inverse=True)
        log_p = F.log_softmax(-dist, dim=-1)[np.arange(len(dist)), test_clsses]

        loss = torch.tensor([0.0], device=self.device)
        for label in classes:
            label_idx = (test_labels == label)
            loss -= log_p[label_idx].sum() / label_idx.sum() / len(classes)

        return loss

    def fit(self, train_dataloader, epochs, lr):
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc='Fitting...', postfix={'epoch': epoch})
            for batch in pbar:
                batch = [el.to(device) for el in batch]
                loss = self.__loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())
                pbar.set_postfix(postfix={'epoch': epoch, 'loss': loss.item()})

        return np.array(losses)

    @torch.no_grad()
    def adapt_few_shots(self, batch, dloader):
        """
          batch: n-way_test k-shot_test batch (pair) of images ([k_shot_test * n-way_test, 1, 28, 28]) \\
                  and labeles [k_shot_test * n-way_test])
          dloader: dataloader for the test set. yields batches of images ([batch_size, 1, 28, 28])\\
                    with their labelel ([batch_size])

          returns pred: np.array of predicted classes for each images in dloader (don't shuffle it)
        """
        self.eval()
        imgs, labels = [el.to(device) for el in batch]
        centers, classes = self.get_clusters(imgs, labels)

        pred = []
        for batch in dloader:
            imgs, labels = [el.to(device) for el in batch]
            pred_idxs = torch.argmin(pairwise_dist(self.net(imgs), centers), dim=-1)
            pred.append(classes[pred_idxs].cpu().numpy())
        pred = np.concatenate(pred)
        return pred
