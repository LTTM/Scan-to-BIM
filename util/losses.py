import torch
from torch import nn
from torch.nn import functional as F

class HNMCrossEntropyLoss(nn.Module):
    def __init__(self, ratio=.2, *args, **kwargs):
        super(HNMCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(*args, reduction='none', **kwargs)
        if not 0 < ratio <= 1:
            raise ValueError("Illegal value %f for the hard mining ratio, must be 0<r<=1"%ratio)
        self.ratio = ratio

    def forward(self, x, y):
        l = self.ce(x, y)
        l = l.reshape(l.shape[0], -1)
        n = int(l.shape[1]*self.ratio)
        with torch.no_grad():
            ids = torch.argsort(l, dim=1, descending=True)[:,:n]
        r = 0
        for b in range(l.shape[0]):
            r += l[b,ids[b]].mean()
        return r/l.shape[0]

class ClassWiseCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClassWiseCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(*args, reduction='none', **kwargs)

    def forward(self, x, y):
        B, C, _, _, _ = x.shape
        l = self.ce(x, y)
        r = torch.zeros(B, device=l.device)
        for b in range(B):
            for c in range(C):
                m = y[b]==c
                if m.any():
                    r[b] += l[b,m].mean()/C
        return r.mean()