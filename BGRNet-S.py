import torch
import os
import torch.nn as nn
# from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
# from model.attention import GlobalContextBlock
from backbone.vgg import B2_VGG
# from backbone.Shunted.SSA import shunted_t
from TLD_BGRNet.Shunted.SSA import shunted_t
import sys
from collections import OrderedDict
import functools

class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode,align_corners=True)
        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimplePP(nn.Module):
    def __init__(self, in_planes, out_planes,k=5):
        super(SimplePP, self).__init__()
        self.conv1 = BasicConv2d(in_planes, in_planes//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_planes*2, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.mp = nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        # print(x.shape)
        x1 = self.mp(x)
        x2 = self.mp(x1)
        # print(x2.shape)
        out = self.conv2(torch.cat((x,x1,x2,self.mp(x2)),dim=1))
        # print(out.shape)
        return out

'''
MAIN_MODEL!!!!
'''
class SNet(nn.Module):
    def __init__(self, ):
        super(SNet, self).__init__()
        # self.vgg = B2_VGG('rgb')
        # self.vgg_dep = B2_VGG('dep')
        self.backbone = shunted_t()
        ##这个地方加载权重
        load_state_dict = torch.load('D:\TLD\TLD_BGRNet\Shunted\ckpt_T.pth')
        self.backbone.load_state_dict(load_state_dict)

        # self.fusion5 = Fusion1(in_plane1=512)
        # self.fusion4 = Fusion1(in_plane1=512)
        # self.fusion3 = Fusion1(in_plane1=256)
        # self.fusion2 = Fusion1(in_plane1=128)
        # self.fusion1 = Fusion1(in_plane1=64)
        self.enhance1 = SimplePP(in_planes=512,out_planes=512)
        self.enhance2 = SimplePP(256,256)
        # self.enhance1 = SimplePP(in_planes=512, out_planes=512)
        # self.enhance2 = SimplePP(256, 256)
        self.enhance3 = nn.Sequential(BasicConv2d(128,128,3,1,1),BasicConv2d(128,128,3,1,1))
        self.enhance4 = nn.Sequential(BasicConv2d(64,64,3,1,1),
                                      BasicConv2d(64,64,3,1,1))
        self.conv64_3 = BasicConv2d(64,1,3,1,1)
        self.conv1_3 = BasicConv2d(1,3,3,1,1)
        self.upsample2 = Up(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = Up(scale_factor=4, mode='bilinear', align_corners=True)
        self.Merge_out1 = Merge_out(in1=256,in2=512)
        self.Merge_out2 = Merge_out(in1=128,in2=256)
        self.Merge_out3 = Merge_out(in1=64,in2=128)
        #11111111111111
        self.conv512_64 = BasicConv2d(512,64,3,1,1)


    def forward(self, x, y):
        rgb = self.backbone(x)
        # print(rgb[0].shape)

        merges = []

        # for i in range(2):
        #     merges.append(rgb[i])
        merges.append(self.enhance4(rgb[0]))
        merges.append(self.enhance3(rgb[1]))
        merges.append(self.enhance2(rgb[2]))
        merges.append(self.enhance1(rgb[3]))

        merge_out1 = self.Merge_out1(merges[-2],merges[-1])
        merge_out2 = self.Merge_out2(merges[-3],merge_out1)
        merge_out3 = self.Merge_out3(merges[-4],merge_out2)
        # merge_out3 = self.attention3(merge_out3)
        # merge_out4 = self.Merge_out4(merges[0],merge_out3)
        # out = self.Decoder(merge_out4,merge_final)
        # print(merge_out3.shape)
        # print(merge_out2.shape)
        # print(merge_out1.shape)
        # merge_out3 = self.upsample2(self.conv128_64(merge_out3))
        # merge_out2 = self.upsample4(self.conv256_64(merge_out2))
        # merge_out1 = self.upsample2(self.conv512_64(self.upsample4(merge_out1)))

        #64,80,120
        out = self.conv64_3(self.upsample4(merge_out3))
        #11111111111111


        return out,merges[3],merges[2],merges[1],merges[0]


class Merge_out(nn.Module):
    def __init__(self, in1,in2):
        super(Merge_out,self).__init__()
        self.bcon2 = BasicConv2d(in_planes=in2,out_planes=in1,kernel_size=1,stride=1,padding=0)
        self.bconv = BasicConv2d(in_planes=in1*2,out_planes=in1 ,kernel_size=1,stride=1,padding=0)
        self.upsample2 = Up(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,d1,d5):
        d5 = self.bcon2(d5)
        d5 = self.upsample2(d5)
        out = torch.cat((d1,d5),dim=1)
        out = self.bconv(out)
        # out = self.mp(out)
        return out

class AMLP(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(AMLP, self).__init__()
        self.fc1 = nn.Linear(1,in_planes//16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_planes//16,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(x)))
        out = avg_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super(SpatialAttention, self).__init__()
        self.conv_s = nn.Conv2d(2,1,kernel,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv_s(x)
        out = self.sigmoid(x)
        return out

class Fusion1(nn.Module):
    def __init__(self, in_plane1,reduction=16, bn_momentum=0.0003):
        self.init__ = super(Fusion1, self).__init__()
        self.conv = BasicConv2d(in_plane1,in_plane1,kernel_size=3,stride=1,padding=1)
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.mlp = AMLP(in_plane1)
        self.conv_mask = nn.Conv2d(in_plane1, 1, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)  # -> b c h*w
        input_x = input_x.unsqueeze(1)  # -> b 1 c hw
        x = self.conv1(x)
        res_x = self.conv_mask(x)  # b 1 h w
        res_x = res_x.view(batch, 1, height * width)  # b 1 hw
        res_x = self.softmax(res_x)  # b 1 hw
        res_x = res_x.unsqueeze(-1)  # b 1 hw 1
        context = torch.matmul(input_x, res_x)  # b(1 c hw  *  1 hw 1) -> b 1 c 1
        context = context.view(batch, channel, 1, 1)  # b c 1 1

        return context

    def forward(self, x):
        _,C,H,W = x.shape
        feature_s = self.spatial_pool(x)
        # feature_s = self.gp(feature_s)
        rgb_assist,dep_assist = feature_s[:,0:C,:,:],feature_s[:,C:2*C,:,:]
        rgb_assist = self.attemp(rgb_assist)
        dep_assist = self.attemp(dep_assist)
        out = self.mlp(rgb_assist+dep_assist)

        return out


if __name__ == '__main__':
    model = SNet()
    left = torch.randn(6, 3, 320, 320)
    right = torch.randn(6, 1, 320, 320)
    out = model(left,right)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)
