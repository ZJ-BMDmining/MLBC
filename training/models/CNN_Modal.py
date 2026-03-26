from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from SENet import SENet
from .ResNet import *

class _CNN_Bone(nn.Module):
    def __init__(self):
        super(_CNN_Bone, self).__init__()
        num ,p= 20,0
        self.block1 = ConvLayer(1, num, (7, 2, 0), (3, 2, 0), p)
        self.block2 = ConvLayer(num, 2*num, (4, 1, 0), (2, 2, 0), p)
        self.block3 = ConvLayer(2*num, 4*num, (3, 1, 0), (2, 2, 0), p)
        self.block4 = ConvLayer(4*num, 8*num, (3, 1, 0), (2, 2, 0), p)
        self.size = self.test_size()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x
    def test_size(self):
        case = torch.ones((1, 1, 182, 218, 182))
        output = self.forward(case)
        return output.shape[1]
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size ,num_classes=10):  # if binary out_size=2; trinary out_size=3
        super(MLP, self).__init__()
        fil_num, drop_rate= 128, 0
        self.fil_num = fil_num
        self.out_size = num_classes
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, num_classes),
        )

    def forward(self, x, get_intermediate_score=False):
        x = self.dense1(x)
        if get_intermediate_score:
            return x
        x = self.dense2(x)
        return x
class CNN_Model(nn.Module):
    def __init__(self,num_classes):
        super(CNN_Model, self).__init__()
        self.backbone = _CNN_Bone()
        # self.backbone = resnet18()
        # self.backbone = DenseNet(growth_rate=6, block_config=(2, 2, 2), compression=0.5,
        #          num_init_features=10, bn_size=2, drop_rate=0, efficient=False)
        # self.backbone = SENet()
        # self.backbone = Multi_SENet()
        self.mlp = MLP(self.backbone.size,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        x = self.mlp(x)
        x = self.softmax(x)
        return x
    
    def get_feature(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.backbone.size)
        return x