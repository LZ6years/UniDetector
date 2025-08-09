from .rpn_head import RPNHead
from .bbox_head import BBoxHead
from .clip_bbox_head import CLIPBBoxHead, CLIPFeatureExtractor
from .bbox_head_clip_inference import BBoxHeadCLIPInference

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../..'))
from mmdet.models.builder import HEADS

def build_head(cfg):
    """构建头部网络"""
    return HEADS.build(cfg)

__all__ = [
    'RPNHead', 
    'BBoxHead', 
    'CLIPBBoxHead',
    'CLIPFeatureExtractor',
    'BBoxHeadCLIPInference',
    'build_head'
] 