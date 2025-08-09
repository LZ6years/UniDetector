# 暂时为空，后续可以添加颈部网络模块

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../..'))
from mmdet.models.builder import NECKS

def build_neck(cfg):
    """构建颈部网络"""
    return NECKS.build(cfg)

__all__ = ['build_neck']
