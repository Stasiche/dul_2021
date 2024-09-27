from src.model import CNF
import numpy as np
import torch


class HCNF(CNF):
    def get_dlogpz_dt(self, f, z):
        v = 2 * torch.randint(0, 2, z.shape, dtype=torch.float, device=self.device) - 1
        A = torch.autograd.grad((v * f).sum(dim=(1, 0)), z, create_graph=True)[0]
        trace = (A * v).sum(dim=1)

        return -trace.reshape(z.shape[0], 1)
