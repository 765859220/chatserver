from einops import rearrange
from torch import nn


class Rearrange(nn.Module):
    def forward(self, x, *args, **kwargs):
        return rearrange(x, *args, **kwargs)
