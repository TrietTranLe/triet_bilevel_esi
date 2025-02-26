import einops
import torch.nn.functional as F
from torch import nn


# define the different cost functions
class CosineReshape(nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, x, y):
        return (1 - F.cosine_similarity(einops.rearrange(x, 'b c h -> b (c h)'), einops.rearrange(y, 'b c h -> b (c h)'))).mean()


class Cosine(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x, y):
        return (1 - F.cosine_similarity(x, y, dim=1) ).mean() # mean over batch of cosine similarity for each time point