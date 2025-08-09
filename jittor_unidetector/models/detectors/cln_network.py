import jittor as jt
import jittor.nn as nn
import numpy as np
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import DETECTORS
from models.backbones.clip_backbone import CLIPResNet
from models.heads.rpn_head import RPNHead

@DETECTORS.register_module(force=True)
class CLNNetwork(nn.Module):
    """
    CLN (Class-agnostic Network) 类无关网络
    专注于目标定位，不依赖类别信息
    """
    
    def __init__(self, backbone_cfg, rpn_cfg, train_cfg=None, test_cfg=None):
        super(CLNNetwork, self).__init__()
        
        # 骨干网络
        self.backbone = CLIPResNet(**backbone_cfg)
        
        # RPN头
        self.rpn_head = RPNHead(**rpn_cfg)
        
        # 训练和测试配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        self.backbone.init_weights()
        self.rpn_head.init_weights()
    
    def extract_feat(self, img):
        """
        提取特征
        Args:
            img: 输入图像 [B, 3, H, W]
        Returns:
            tuple: 多尺度特征
        """
        return self.backbone(img)
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        """
        训练时前向传播
        Args:
            img: 输入图像
            img_metas: 图像元信息
            gt_bboxes: 真实边界框
            gt_labels: 真实标签
        Returns:
            dict: 损失字典
        """
        # 提取特征
        feat = self.extract_feat(img)
        
        # RPN前向传播
        rpn_outs = self.rpn_head(feat)
        
        # 计算损失
        losses = self.rpn_head.loss(*rpn_outs, gt_bboxes, gt_labels, img_metas)
        
        return losses
    
    def forward_test(self, img, img_metas):
        """
        测试时前向传播
        Args:
            img: 输入图像
            img_metas: 图像元信息
        Returns:
            list: 检测结果
        """
        # 提取特征
        feat = self.extract_feat(img)
        
        # RPN前向传播
        rpn_outs = self.rpn_head(feat)
        
        # 生成提议
        proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas)
        
        return proposal_list
    
    def execute(self, img, mode='test', **kwargs):
        """
        统一的前向传播接口
        Args:
            img: 输入图像
            mode: 模式 ('train' or 'test')
            **kwargs: 其他参数
        Returns:
            根据模式返回不同结果
        """
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')


class CLNProposalGenerator:
    """
    CLN提议生成器
    基于CLN网络生成区域提议
    """
    
    def __init__(self, model, nms_thresh=0.7, n_pre_nms=12000, n_post_nms=2000):
        self.model = model
        self.nms_thresh = nms_thresh
        self.n_pre_nms = n_pre_nms
        self.n_post_nms = n_post_nms
    
    def generate_proposals(self, img, img_metas):
        """
        生成区域提议
        Args:
            img: 输入图像
            img_metas: 图像元信息
        Returns:
            proposals: 区域提议列表
        """
        # 使用CLN网络生成提议
        proposals = self.model.forward_test(img, img_metas)
        
        return proposals
    
    def filter_proposals(self, proposals, score_thresh=0.05):
        """
        过滤提议
        Args:
            proposals: 原始提议
            score_thresh: 分数阈值
        Returns:
            filtered_proposals: 过滤后的提议
        """
        filtered_proposals = []
        
        for proposal in proposals:
            # 过滤低分数提议
            keep = proposal[:, 4] > score_thresh
            filtered_proposal = proposal[keep]
            
            # 限制提议数量
            if len(filtered_proposal) > self.n_post_nms:
                # 按分数排序
                scores = filtered_proposal[:, 4]
                indices = jt.argsort(scores, descending=True)[:self.n_post_nms]
                filtered_proposal = filtered_proposal[indices]
            
            filtered_proposals.append(filtered_proposal)
        
        return filtered_proposals 