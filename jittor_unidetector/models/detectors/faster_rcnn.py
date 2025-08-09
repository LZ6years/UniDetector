import jittor as jt
import jittor.nn as nn
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import DETECTORS
from .base import BaseDetector

@DETECTORS.register_module(force=True)
class FasterRCNN(BaseDetector):
    """
    纯Jittor版本的FasterRCNN检测器
    不依赖MMDetection，完全基于Jittor实现
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__()
        
        # 构建骨干网络
        self.backbone = self._build_backbone(backbone)
        
        # 构建颈部网络
        if neck is not None:
            self.neck = self._build_neck(neck)
        else:
            self.neck = None
        
        # 构建RPN头
        if rpn_head is not None:
            self.rpn_head = self._build_rpn_head(rpn_head)
        else:
            self.rpn_head = None
        
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
        from models.backbones import build_backbone
        return build_backbone(cfg)
    
    def _build_neck(self, cfg):
        """构建颈部网络"""
        from models.necks import build_neck
        return build_neck(cfg)
    
    def _build_rpn_head(self, cfg):
        """构建RPN头"""
        from models.heads import build_head
        return build_head(cfg)
    
    def _build_roi_head(self, cfg):
        """构建RoI头"""
        from models.roi_heads import build_head
        return build_head(cfg)
    
    def init_weights(self, pretrained=None):
        """初始化权重"""
        if pretrained is not None:
            self.backbone.init_weights(pretrained)
    
    def extract_feat(self, img):
        """提取特征"""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        """训练时前向传播"""
        x = self.extract_feat(img)
        
        losses = dict()
        
        # RPN前向传播和损失计算
        if self.rpn_head is not None:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            
            # 生成proposals
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals
        
        # RoI前向传播和损失计算
        if self.roi_head is not None:
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
            losses.update(roi_losses)
        
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """简单测试"""
        x = self.extract_feat(img)
        
        # 获取proposals
        if proposals is None and self.rpn_head is not None:
            rpn_outs = self.rpn_head(x)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas, cfg=self.test_cfg.rpn)
        else:
            proposal_list = proposals
        
        # RoI测试
        if self.roi_head is not None:
            bbox_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
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