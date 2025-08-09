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


@BACKBONES.register_module(force=True)
class CLIPResNetFPN(nn.Module):
    """
    CLIP ResNet + FPN骨干网络
    包含多尺度特征提取和特征金字塔网络融合
    """
    
    def __init__(self, layers=[3, 4, 6, 3], out_channels=256, style='pytorch', **kwargs):
        super(CLIPResNetFPN, self).__init__()
        
        # 加载CLIP模型
        self.clip_model, _ = clip.load("RN50", device="cpu")
        self.visual = self.clip_model.visual
        
        # 提取各个阶段的特征
        self.conv1 = self.visual.conv1
        self.bn1 = self.visual.bn1
        self.relu = self.visual.relu
        self.maxpool = self.visual.maxpool
        
        # ResNet层
        self.layer1 = self.visual.layer1  # 1/4, 256通道
        self.layer2 = self.visual.layer2  # 1/8, 512通道
        self.layer3 = self.visual.layer3  # 1/16, 1024通道
        self.layer4 = self.visual.layer4  # 1/32, 2048通道
        
        self.layers = layers
        self.style = style
        self.out_channels = out_channels
        
        # FPN lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),   # C2 -> 256
            nn.Conv2d(512, out_channels, 1),   # C3 -> 256
            nn.Conv2d(1024, out_channels, 1),  # C4 -> 256
            nn.Conv2d(2048, out_channels, 1),  # C5 -> 256
        ])
        
        # FPN output convolutions (3x3 conv for final features)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # P2
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # P3
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # P4
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # P5
        ])
        
        # 额外的P6层（用于大目标检测）
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # P6
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # P7
        ])
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化FPN权重"""
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
                    
        for m in self.fpn_convs.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
                    
        for m in self.downsample_convs.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
        
    def execute(self, x):
        """
        前向传播，返回FPN多尺度特征
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            tuple: FPN多尺度特征 (P2, P3, P4, P5, P6, P7)
                  P2: 1/4, P3: 1/8, P4: 1/16, P5: 1/32, P6: 1/64, P7: 1/128
        """
        # 提取ResNet特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)   # 1/4, 256
        c3 = self.layer2(c2)  # 1/8, 512
        c4 = self.layer3(c3)  # 1/16, 1024
        c5 = self.layer4(c4)  # 1/32, 2048
        
        # FPN自顶向下路径
        # 从最高层开始，逐步向下融合
        p5 = self.lateral_convs[3](c5)  # 2048 -> 256
        p4 = self._upsample_add(p5, self.lateral_convs[2](c4))  # 1024 -> 256
        p3 = self._upsample_add(p4, self.lateral_convs[1](c3))  # 512 -> 256
        p2 = self._upsample_add(p3, self.lateral_convs[0](c2))  # 256 -> 256
        
        # 应用3x3卷积得到最终特征
        p2 = self.fpn_convs[0](p2)
        p3 = self.fpn_convs[1](p3)
        p4 = self.fpn_convs[2](p4)
        p5 = self.fpn_convs[3](p5)
        
        # 额外的下采样层
        p6 = self.downsample_convs[0](p5)  # 1/64
        p7 = self.downsample_convs[1](p6)  # 1/128
        
        return (p2, p3, p4, p5, p6, p7)
    
    def _upsample_add(self, x, y):
        """
        上采样并相加
        Args:
            x: 高层特征
            y: 低层特征
        Returns:
            融合后的特征
        """
        # 上采样x到y的尺寸
        _, _, H, W = y.shape
        x_upsampled = jt.nn.interpolate(x, size=(H, W), mode='nearest')
        return x_upsampled + y
    
    def init_weights(self, pretrained=None):
        """初始化权重"""
        if pretrained is not None:
            # 加载预训练权重
            pass 