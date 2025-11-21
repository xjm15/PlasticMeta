import torch
from torch.optim import Adam


class CustomAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(CustomAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def step(self, closure=None):
        return super(CustomAdam, self).step(closure)