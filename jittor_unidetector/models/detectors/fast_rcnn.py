import jittor as jt
import jittor.nn as nn
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import DETECTORS
from .base import BaseDetector

@DETECTORS.register_module(force=True)
class FastRCNN(BaseDetector):
    """
    纯Jittor版本的FastRCNN检测器
    用于第二阶段训练，使用预生成的proposals
    """
    
    def __init__(self,
                 backbone,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FastRCNN, self).__init__()
        
        # 构建骨干网络
        self.backbone = self._build_backbone(backbone)
        
        # 构建RoI头
        if roi_head is not None:
            self.roi_head = self._build_roi_head(roi_head)
        else:
            self.roi_head = None
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 初始化权重
        self.init_weights(pretrained)
    
    def _build_backbone(self, cfg):
        """构建骨干网络"""
        from ..backbones import build_backbone
        return build_backbone(cfg)
    
    def _build_roi_head(self, cfg):
        """构建RoI头"""
        from ..heads.roi_heads import build_head
        return build_head(cfg)
    
    def init_weights(self, pretrained=None):
        """初始化权重"""
        if pretrained is not None:
            self.backbone.init_weights(pretrained)
    
    def extract_feat(self, img):
        """提取特征"""
        x = self.backbone(img)
        return x
    
    def forward_train(self, img, img_metas, proposals, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, **kwargs):
        """训练时前向传播"""
        x = self.extract_feat(img)
        
        losses = dict()
        
        # RoI前向传播和损失计算
        if self.roi_head is not None:
            roi_losses = self.roi_head.forward_train(x, img_metas, proposals, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
            losses.update(roi_losses)
        
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """简单测试"""
        x = self.extract_feat(img)
        
        # RoI测试
        if self.roi_head is not None:
            bbox_results = self.roi_head.simple_test(x, proposals, img_metas, rescale=rescale)
            return bbox_results
        else:
            return None
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """增强测试"""
        raise NotImplementedError
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """测试时前向传播"""
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != num of image meta ({len(img_metas)})')
        
        # 简单测试
        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)
    
    def forward(self, img, mode='forward', **kwargs):
        """统一前向传播接口"""
        if mode == 'forward':
            return self.forward_test(img, **kwargs)
        elif mode == 'loss':
            return self.forward_train(img, **kwargs)
        else:
            raise ValueError(f'Unknown mode {mode}') 