from .oln_roi_head import OlnRoIHead

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../..'))
from mmdet.models.builder import HEADS

def build_head(cfg):
    """构建RoI头部网络"""
    return HEADS.build(cfg)

__all__ = ['OlnRoIHead', 'build_head']
