from .clip_backbone import CLIPResNet
# from .resnet import ResNet  # 暂时注释

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))
from mmdet.models.builder import BACKBONES

def build_backbone(cfg):
    """构建骨干网络"""
    return BACKBONES.build(cfg)

__all__ = ['CLIPResNet', 'build_backbone'] 