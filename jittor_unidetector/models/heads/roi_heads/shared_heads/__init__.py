from .clip_res_layer import CLIPResLayer

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import SHARED_HEADS

def build_shared_head(cfg):
    """构建共享头部网络"""
    return SHARED_HEADS.build(cfg)

__all__ = ['CLIPResLayer', 'build_shared_head']
