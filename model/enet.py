# this is a 3D extension to the ENet 2D segmentation architecture
# https://arxiv.org/abs/1606.02147
import torch
from torch import nn

class InputEmbedding(nn.Module):
    def __init__(self, in_chs=1, out_chs=16):
        super(InputEmbedding, self).__init__()
        assert in_chs < out_chs, "Input channels must be less than output channels"
        self.conv = nn.Conv3d(in_chs, out_chs-in_chs, 5, dilation=3, stride=2, padding=6, bias=False)
        self.pool = nn.MaxPool3d(5, stride=2, padding=2)

    def forward(self, x):
        xc = self.conv(x)
        xp = self.pool(x)
        return torch.cat([xc,xp], dim=1)

class ResidualLayer(nn.Module):
    def __init__(self, in_chs, out_chs, reduce=False):
        super(ResidualLayer, self).__init__()
        
        if in_chs != out_chs:
            self.expand = True
            self.conv0 = nn.Conv3d(in_chs, out_chs, 1, bias=False)
        else:
            self.expand = False
        
        self.reduce = reduce
        if reduce:
            self.conv1 = nn.Conv3d(out_chs, out_chs//2, 5, dilation=3, stride=2, padding=6, bias=False)
        else:
            self.conv1 = nn.Conv3d(out_chs, out_chs//2, 1, bias=False)
        
        self.prelu1 = nn.PReLU(out_chs//2)
        self.conv2 = nn.Conv3d(out_chs//2, out_chs//2, 5, dilation=3, padding=6, bias=False)
        self.prelu2 = nn.PReLU(out_chs//2)
        self.conv3 = nn.Conv3d(out_chs//2, out_chs, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_chs)
        self.prelu3 = nn.PReLU(out_chs)
        
        self.pool = nn.MaxPool3d(2)

    def forward(self, x0):
    
        if self.expand:
            x0 = self.conv0(x0)
    
        x = self.conv1(x0)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn(x)
        
        if self.reduce:
            x0 = self.pool(x0)

        x = self.prelu3(x0+x)
        
        return x