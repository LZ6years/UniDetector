import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))
from mmdet.models.builder import DETECTORS, HEADS, BACKBONES, NECKS, SHARED_HEADS

# 重新导出注册表
__all__ = ['DETECTORS', 'HEADS', 'BACKBONES', 'NECKS', 'SHARED_HEADS']
