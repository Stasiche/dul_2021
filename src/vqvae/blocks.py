import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2d(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock2d, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return x + self.model(x)


class VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=1):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.beta = beta

    def forward(self, z):
        n, _, h, w = z.shape
        with torch.no_grad():
            tmp = z.permute(0, 2, 3, 1).reshape(-1, self.embedding.embedding_dim)

            distances = torch.sum(torch.pow(tmp[:, None, :] - self.embedding.weight[None, :, :], 2), 2)
            indxs = torch.argmin(distances, dim=1).reshape(n, h, w)

        quantized = self.embedding(indxs).permute(0, 3, 1, 2)

        return quantized, z + (quantized - z).detach(), indxs

    def loss(self, embedding, q_embedding):
        encoder_loss = F.mse_loss(embedding.detach(), q_embedding)
        decoder_loss = F.mse_loss(embedding, q_embedding.detach())

        return encoder_loss + self.beta * decoder_loss
