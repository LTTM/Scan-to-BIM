import torch
from torch import nn
from torch.nn import functional as F

from model.enet import InputEmbedding, ResidualLayer

class SegCloud(nn.Module):
    def __init__(self, inchs=1, num_classes=8, midchs=64):
        super(SegCloud, self).__init__()
        
        self.layer0 = InputEmbedding(inchs, midchs) # stride 2 - segcloud 7x7 in + pool
        
        self.layer1 = ResidualLayer(midchs, 128)
        self.layer2 = ResidualLayer(128, 128, reduce=True) # stride 4 - 1st segcloud layer + pool
        
        self.layer3 = ResidualLayer(128, 256)
        self.layer4 = ResidualLayer(256, 256) # 2nd segcloud layer
        
        self.layer5 = ResidualLayer(256, 256)
        self.layer6 = ResidualLayer(256, 256) # 3rd segcloud layer
        
        self.clas = nn.Conv3d(256, num_classes, 3, padding=1, bias=False)
        self.pool = nn.MaxPool3d(2)
        
        self.x01 = nn.Conv3d(64, 128, 1, bias=False)
        self.x02 = nn.Conv3d(64, 256, 1, bias=False)
        self.x12 = nn.Conv3d(128, 256, 1, bias=False)

    def forward(self, x):

        x0 = self.layer0(x)
        
        x = self.layer1(x0)        
        x1 = self.layer2(x)
        
        x0 = self.pool(x0)
        x01 = self.x01(x0)
        
        x = self.layer3(x1+x01)
        x2 = self.layer4(x)
        
        x02 = self.x02(x0)
        x12 = self.x12(x1)
        
        x = self.layer5(x2+x12+x02)
        x = self.layer6(x)
        
        x = self.clas(x+x2+x12+x02)

        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=True)

        return x
