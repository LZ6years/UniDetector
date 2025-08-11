from .shared2fc_bbox_score_head import Shared2FCBBoxScoreHead
from .bbox_head_clip_inference import BBoxHeadCLIPInference
from .bbox_head_clip_partitioned import BBoxHeadCLIPPartitioned

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS

def build_head(cfg):
    """构建边界框头部网络"""
    return HEADS.build(cfg)

__all__ = [
    'Shared2FCBBoxScoreHead',
    'BBoxHeadCLIPInference', 
    'BBoxHeadCLIPPartitioned',
    'build_head'
]
