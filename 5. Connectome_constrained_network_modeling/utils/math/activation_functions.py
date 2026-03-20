import numpy as np
import torch


def relu(x):
    return np.maximum(x, 0)

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self, c=np.log(2), beta=1.0):
        super().__init__()
        self.c = c
        self.softplus = torch.nn.Softplus(beta=beta)

    def forward(self, x):
        return self.softplus(x) - self.c