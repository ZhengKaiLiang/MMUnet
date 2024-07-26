from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchsummary import summary

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        return x * y


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d, padding_mode='reflect',dilation=d,stride=stride,groups=groups,bias=bias))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        x=torch.tensor_split(x,self.num_splits,dim=1)
        res=[]
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res,dim=1)


class ConvBnActConv(nn.Module):
    def __init__(self,w,stride,dilation,groups,bias):
        super().__init__()
        self.conv=ConvBnAct(w,w,3,stride,dilation,dilation,groups)
        self.project=nn.Conv2d(w,w,1,bias=bias)
    def forward(self,x):
        x=self.conv(x)
        x=self.project(x)
        return x


class MMD(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride=1, attention="se"):
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()

        self.conv_add1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=in_channels, padding=1,
                                   padding_mode='reflect')
        self.bn_add1 = norm2d(out_channels)
        self.act_add1 = activation()

        if len(dilations)==1:
            dilation=dilations[0]
            self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,
                                 padding_mode='reflect',dilation=dilation)
        else:
            self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=True)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3=norm2d(out_channels)
        self.act3=activation()

        if attention=="se":
            self.se=SEModule(out_channels,in_channels//4)
        elif attention=="se2":
            self.se=SEModule(out_channels,out_channels//4)
        else:
            self.se=None

        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)

        x = self.conv_add1(x)
        x = self.bn_add1(x)
        x = self.act_add1(x)
        residual_add1 = x

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        residual_d = x

        if self.se is not None:
            x=self.se(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut + residual_add1 + residual_d)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, group_width, layer_num=1, dilations=[]):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(MMD(in_channels=out_channels, out_channels=out_channels, dilations=dilations, group_width= group_width))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.squeeze = nn.Conv2d(dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class ECA(nn.Module):
    def __init__(self,dim,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(dim,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, group_width, bilinear=True, layer_num=1, dilations=[]):
        super(Up, self).__init__()
        C = in_channels // 2
        self.norm = nn.BatchNorm2d(C)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(MMD(in_channels=out_channels, out_channels=out_channels, dilations=dilations, group_width= group_width))
        self.conv = nn.Sequential(*layers)
        self.sa = SpatialAttention(C)
        self.eca = ECA(C)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        fsa = self.sa(x2)
        feca = self.eca(x2)
        x2 = x2+(fsa * feca)

        # B, _, H, W = x2.shape
        # x2 = x2.flatten(2).transpose(1, 2)
        # x2 = self.shifted(x2, H, W)
        # x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class MMUnet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear=True,
                 base_c: int = 32):
        super(MMUnet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.ReLU(),
            MMD(in_channels=base_c, out_channels=base_c, dilations=[1,1], group_width=base_c//4, stride=1)
        )
        self.down1 = Down(base_c, base_c * 2, group_width=base_c * 2//4, dilations=[1,2])
        self.down2 = Down(base_c * 2, base_c * 4, group_width=base_c * 4//4, dilations=[1,3])
        self.down3 = Down(base_c * 4, base_c * 8, group_width=base_c * 8//4, dilations=[2,3])
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16//factor, group_width=base_c * 16//4, dilations=[2,7])

        self.up1 = Up(base_c * 16, base_c * 8 // factor, group_width=base_c * 8 // factor//4, bilinear=bilinear, dilations=[2,9])
        self.up2 = Up(base_c * 8, base_c * 4 // factor, group_width=base_c * 4 // factor//4, bilinear=bilinear, dilations=[2,11])
        self.up3 = Up(base_c * 4, base_c * 2 // factor, group_width=base_c * 2 // factor//4, bilinear=bilinear, dilations=[4,7])
        self.up4 = Up(base_c * 2, base_c, group_width=base_c//4, bilinear=bilinear, dilations=[5,14])
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        return logits


if __name__ == '__main__':
    model = MMUnet(in_channels=3, num_classes=1, base_c=32).to('cuda')
    input = torch.randn(1, 3, 256, 256).to('cuda')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')