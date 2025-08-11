import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../..'))
from mmdet.models.builder import HEADS

# 先导入基础模块
from .bbox_head import BBoxHead
from .rpn_head import RPNHead
from .clip_bbox_head import CLIPBBoxHead, CLIPFeatureExtractor
from .oln_rpn_head import OlnRPNHead

# 再导入roi_heads下的模块
from .roi_heads.oln_roi_head import OlnRoIHead
from .roi_heads.bbox_heads.bbox_head_clip_partitioned import BBoxHeadCLIPPartitioned
from .roi_heads.bbox_heads.bbox_head_clip_inference import BBoxHeadCLIPInference
from .roi_heads.bbox_heads.shared2fc_bbox_score_head import Shared2FCBBoxScoreHead
from .roi_heads.shared_heads.clip_res_layer import CLIPResLayer

def build_head(cfg):
    """构建头部网络"""
    return HEADS.build(cfg)

__all__ = [
    'RPNHead', 
    'BBoxHead', 
    'CLIPBBoxHead',
    'CLIPFeatureExtractor',
    'OlnRPNHead',
    'OlnRoIHead',
    'BBoxHeadCLIPPartitioned',
    'BBoxHeadCLIPInference',
    'Shared2FCBBoxScoreHead',
    'CLIPResLayer',
    'build_head'
] 