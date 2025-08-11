import jittor as jt
import jittor.nn as nn
import clip
import numpy as np
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import BACKBONES

class Bottleneck(nn.Module):
    """Bottleneck block for CLIP ResNet"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def execute(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

@BACKBONES.register_module(force=True)
class CLIPResNet(nn.Module):
    """
    CLIP ResNet骨干网络，参考UniDetector实现
    基于Jittor框架重写
    """
    
    def __init__(self, layers=[3, 4, 6, 3], output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        
        self.frozen_stages = 1
        self._freeze_stages()

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            # 加载CLIP预训练权重
            checkpoint = jt.load(pretrained)
            
            state_dict = {}
            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            # 加载权重
            self.load_state_dict(state_dict)
            print('CLIPResNet weights loaded from:', pretrained)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)

        outs = []
        x = self.layer1(x)
        # outs.append(x)
        x = self.layer2(x)
        # outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        # x = self.layer4(x)
        # outs.append(x)
        return tuple(outs)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.layer1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer freezed."""
        super(CLIPResNet, self).train(mode)
        self._freeze_stages()
        for m in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, nn.BatchNorm2d):
                m.eval()