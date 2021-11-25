import torch.nn as nn


class UWBModel(nn.Module):
    def __init__(self, inp_dim, hidden_dim, width):
        super().__init__()

        self.width = width
        self.inp_dim = inp_dim
        self.bs = inp_dim * width

        self.main = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.bs * 2 + width),
        )

    def forward(self, t):
        uwb = self.main(t.unsqueeze(0)).squeeze(0)

        u = uwb[: self.bs].reshape(self.width, 1, self.inp_dim)
        w = uwb[self.bs: 2 * self.bs].reshape(self.width, self.inp_dim, 1)
        b = uwb[-self.width:].reshape(self.width, 1, 1)

        return u, w, b
