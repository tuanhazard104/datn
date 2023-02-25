# Hà Minh Sơn
import sys
sys.path.append("E:/tai_lieu_hoc_tap/tdh/tuannca_datn")
import torch
from torch.nn import Dropout

import math

from os.path import join as pjoin
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from networks.vit_seg_modeling_resnet_skip import StdConv2d, PreActBottleneck
from torchvision.io import read_image
def flatten(x):
    return x.flatten(2)

def transpose(x):
    return x.transpose(-1, -2)

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor=1):
        # block_units = (3,4,9)
        super().__init__()
        width = int(64 * width_factor)
        self.width = width # 64
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        # print("before root: ", x.size()) # (2,3,256,256)
        x = self.root(x)
        # print("after root: ", x.size()) # (2,64,128,128)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        # print("after maxpool2D: ", x.size()) # [2, 64, 63, 63]
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        # print("after body: ", x.size()) # [2, 1024, 16, 16]
        return x, features[::-1]
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

if __name__ == "__main__":
    # t = torch.randn(2, 3, 3, 3)
    # # print(t)
    # print("before flatten: ",t.size())
    # x = t.flatten(2)
    # print("after flatten: ", x.size())
    # print(x)
    # y1 = x.transpose(-1, -2)
    # y2 = x.transpose(1, 2)
    # print("after transpose: ", y1.size())
    # print("after transpose: ", y2.size())
    # print(y1)
    # print(y2)

    # # dropouter = Dropout(p=0.5)
    # # z = dropouter(y)
    # # print("after dropout: ", z.size())
    # # print(z)

    # x = read_image("E:/tai_lieu_hoc_tap/tdh/tuannca_datn/test.jpg")
    # print(x.size())
    # x = F.conv2d(x, weight=64)
    # print(x.size())
    dim = 512
    expand =  nn.Linear(dim, 2*dim, bias=False)
    up1 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=1,stride=2,padding=1)
    # up2 = nn.UpsamplingBilinear2d(size=dim//2,scale_factor=2)
    x = torch.randn(1, 256, 14, 14)
    print(x.size())
    # up = up_conv(ch_in=512,ch_out=256)

    # x = up(x)
    x = up1(x)
    print(x.size())
