import torch
import torch.nn as nn
import torch.nn.functional as F

# SELayer模块（用于通道注意力机制）
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# MultiScaleConv模块（多尺度卷积模块）
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn(x)
        return x

# new_AttentionMultiScaleCNN模型定义
class new_AttentionMultiScaleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(new_AttentionMultiScaleCNN, self).__init__()
        # 第一层多尺度卷积和SE注意力模块
        self.multi_scale_conv1 = MultiScaleConv(3, 64)   # 输入3通道，输出64通道
        self.se1 = SELayer(64 * 3)                        # SE注意力机制，通道数=64*3

        # 第二层多尺度卷积和SE注意力模块
        self.multi_scale_conv2 = MultiScaleConv(64 * 3, 100)
        self.se2 = SELayer(100 * 3)

        # 第三层多尺度卷积和SE注意力模块
        self.multi_scale_conv3 = MultiScaleConv(100 * 3, 50)
        self.se3 = SELayer(50 * 3)

        # 其他层
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(50 * 3 * 22 * 22, num_classes)

        self.layers = [
            self.multi_scale_conv1,
            self.se1,
            self.relu1,
            self.maxpool1,
            self.multi_scale_conv2,
            self.se2,
            self.relu2,
            self.maxpool2,
            self.multi_scale_conv3,
            self.se3,
            self.relu3,
            self.maxpool3,
            self.flatten,
            self.dropout,
            self.fc
        ]

    def forward(self, x):
        # 第一层卷积、注意力和池化
        x = self.relu1(self.multi_scale_conv1(x))
        x = self.se1(x)
        x = self.maxpool1(x)

        # 第二层卷积、注意力和池化
        x = self.relu2(self.multi_scale_conv2(x))
        x = self.se2(x)
        x = self.maxpool2(x)

        # 第三层卷积、注意力和池化
        x = self.relu3(self.multi_scale_conv3(x))
        x = self.se3(x)
        x = self.maxpool3(x)
        # print(x.shape)
        # 展平、全连接和分类输出
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
