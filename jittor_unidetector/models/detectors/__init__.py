from .faster_rcnn import FasterRCNN

import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../..'))
from mmdet.models.builder import DETECTORS

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """构建检测器"""
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

__all__ = ['FasterRCNN', 'build_detector'] 