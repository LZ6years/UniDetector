import jittor as jt
import jittor.nn as nn
import numpy as np
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS

@HEADS.register_module(force=True)
class RPNHead(nn.Module):
    """
    区域提议网络头 (Region Proposal Network Head)
    用于生成候选区域
    """
    
    def __init__(self, in_channels, feat_channels, num_anchors=3, 
                 anchor_scales=[2, 4, 8, 16, 32], anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_generator=None, assigner=None, sampler=None, 
                 bbox_coder=None, loss_cls=None, loss_bbox=None, 
                 train_cfg=None, test_cfg=None, **kwargs):
        super(RPNHead, self).__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        # 从 anchor_generator 推断每点 anchors 数
        if isinstance(anchor_generator, dict):
            try:
                ag_scales = anchor_generator.get('scales', anchor_scales)
                ag_ratios = anchor_generator.get('ratios', anchor_ratios)
                self.num_anchors = max(1, len(ag_scales) * len(ag_ratios))
                self.anchor_scales = ag_scales
                self.anchor_ratios = ag_ratios
            except Exception:
                self.num_anchors = num_anchors
                self.anchor_scales = anchor_scales
                self.anchor_ratios = anchor_ratios
        else:
            self.num_anchors = num_anchors
            self.anchor_scales = anchor_scales
            self.anchor_ratios = anchor_ratios
        
        # 保存配置参数（暂时不使用，但需要接受）
        self.anchor_generator = anchor_generator
        self.assigner = assigner
        self.sampler = sampler
        self.bbox_coder = bbox_coder
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # RPN卷积层
        self.conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.relu = nn.ReLU()
        
        # 分类头：预测前景/背景
        self.cls_head = nn.Conv2d(feat_channels, self.num_anchors * 2, 1)
        
        # 回归头：预测边界框偏移
        self.reg_head = nn.Conv2d(feat_channels, self.num_anchors * 4, 1)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        # 使用正态分布初始化
        jt.init.gauss_(self.conv.weight, std=0.01)
        jt.init.constant_(self.conv.bias, 0)
        
        jt.init.gauss_(self.cls_head.weight, std=0.01)
        jt.init.constant_(self.cls_head.bias, 0)
        
        jt.init.gauss_(self.reg_head.weight, std=0.01)
        jt.init.constant_(self.reg_head.bias, 0)
    
    def _forward_single(self, feat):
        feat = self.conv(feat)
        feat = self.relu(feat)
        cls_scores = self.cls_head(feat)
        bbox_preds = self.reg_head(feat)
        return cls_scores, bbox_preds

    def execute(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [B, C, H, W] 或 List[Var[B, C, H_i, W_i]]（来自 FPN 多层）
        Returns:
            若输入为单层，返回 tuple(cls_scores, bbox_preds)
            若输入为多层，返回 (list[cls_scores], list[bbox_preds])
        """
        if isinstance(x, (list, tuple)):
            cls_scores_list = []
            bbox_preds_list = []
            for feat in x:
                cls_s, bbox_p = self._forward_single(feat)
                cls_scores_list.append(cls_s)
                bbox_preds_list.append(bbox_p)
            return cls_scores_list, bbox_preds_list
        else:
            return self._forward_single(x)


class AnchorGenerator:
    """
    锚框生成器
    生成不同尺度和比例的锚框
    """
    
    def __init__(self, scales, ratios, strides):
        self.scales = scales
        self.ratios = ratios
        self.strides = strides
        
    def generate_anchors(self, featmap_size, stride):
        """
        生成锚框
        Args:
            featmap_size: 特征图尺寸 (H, W)
            stride: 步长
        Returns:
            anchors: 锚框坐标 [N, 4] (x1, y1, x2, y2)
        """
        anchors = []
        
        for scale in self.scales:
            for ratio in self.ratios:
                # 计算锚框的宽高
                w = scale * np.sqrt(ratio)
                h = scale / np.sqrt(ratio)
                
                # 生成网格点
                for i in range(featmap_size[0]):
                    for j in range(featmap_size[1]):
                        # 计算锚框中心点
                        cx = (j + 0.5) * stride
                        cy = (i + 0.5) * stride
                        
                        # 计算锚框坐标
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        
                        anchors.append([x1, y1, x2, y2])
        
        return np.array(anchors) 