import jittor as jt
import jittor.nn as nn
import numpy as np
import os
import json
import sys

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS

@HEADS.register_module(force=True)
class BBoxHeadCLIPInference(nn.Module):
    """
    CLIP BBox Head for inference with probability calibration
    基于UniDetector的BBoxHeadCLIPInference实现
    """
    
    def __init__(self, in_channels, num_classes, roi_feat_size=7,
                 clip_embedding_path=None, temperature=0.07,
                 beta=0.3, with_calibration=False, gamma=0.6,
                 result_file=None, prior_prob_path=None, **kwargs):
        super(BBoxHeadCLIPInference, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.roi_feat_size = roi_feat_size
        self.temperature = temperature
        self.beta = beta
        self.with_calibration = with_calibration
        self.gamma = gamma
        
        # RoI特征提取
        self.avg_pool = nn.AdaptiveAvgPool2d((roi_feat_size, roi_feat_size))
        
        # 特征投影层
        self.fc = nn.Linear(in_channels * roi_feat_size * roi_feat_size, 1024)
        self.relu = nn.ReLU()
        
        # 分类和回归头
        self.cls_head = nn.Linear(1024, num_classes)
        self.reg_head = nn.Linear(1024, num_classes * 4)
        
        # 加载CLIP文本嵌入
        self.clip_embeddings = self.load_clip_embeddings(clip_embedding_path)
        
        # 加载检测结果统计 (用于频率校准)
        self.cnum = None
        if with_calibration and result_file:
            self.cnum = self.load_detection_statistics(result_file)
        
        # 加载先验概率
        self.prior_probs = None
        if prior_prob_path and os.path.exists(prior_prob_path):
            self.prior_probs = np.load(prior_prob_path)
            print(f"Loaded prior probabilities from {prior_prob_path}")
        
        # 初始化权重
        self.init_weights()
    
    def load_clip_embeddings(self, embedding_path):
        """加载CLIP文本嵌入"""
        if embedding_path and os.path.exists(embedding_path):
            embeddings = np.load(embedding_path)
        else:
            # 如果没有预计算的嵌入，使用随机初始化
            embeddings = np.random.randn(self.num_classes, 512).astype(np.float32)
        
        return jt.array(embeddings)
    
    def load_detection_statistics(self, result_file):
        """
        加载检测结果统计
        基于UniDetector的实现，统计每个类别的检测数量
        """
        print(f"Loading detection statistics from {result_file}")
        
        if not os.path.exists(result_file):
            print(f"Warning: Result file not found: {result_file}")
            return None
        
        # 加载预检测结果
        with open(result_file, 'rb') as f:
            preresult = np.load(f, allow_pickle=True)
        
        # 统计每个类别的检测数量
        cnum = np.zeros(self.num_classes)
        for i in range(len(preresult)):
            for nc in range(len(preresult[i])):
                if nc < self.num_classes:
                    cnum[nc] += preresult[i][nc].shape[0]
        
        print(f"Detection statistics: {cnum}")
        return cnum
    
    def init_weights(self):
        """初始化权重"""
        jt.init.gauss_(self.fc.weight, std=0.01)
        jt.init.constant_(self.fc.bias, 0)
        jt.init.gauss_(self.cls_head.weight, std=0.01)
        jt.init.constant_(self.cls_head.bias, 0)
        jt.init.gauss_(self.reg_head.weight, std=0.01)
        jt.init.constant_(self.reg_head.bias, 0)
    
    def calibrate_scores(self, scores, proposal_scores=None):
        """
        概率校准 - 基于UniDetector原始实现
        使用公式: scores = scores^beta * proposal_score^(1-beta)
        频率校准: scores = scores * (1/frequencies^gamma) / mean(frequencies)
        """
        if not self.with_calibration:
            return scores
        
        # 1. 频率校准 (基于检测结果统计)
        if self.cnum is not None:
            frequencies = jt.array(self.cnum, dtype=scores.dtype).view(1, -1)
            # 频率转换: 1 / frequencies^gamma
            frequencies = 1 / (frequencies ** self.gamma)
            # 应用频率权重并归一化
            scores = scores * frequencies / frequencies.mean()
        
        # 2. 提议分数融合 (如果有提议分数)
        if proposal_scores is not None:
            # 确保维度匹配
            if proposal_scores.ndim == 1:
                proposal_scores = proposal_scores.unsqueeze(1)  # [B, 1]
            # 融合公式: scores^beta * proposal_score^(1-beta)
            scores = (scores ** self.beta) * (proposal_scores ** (1 - self.beta))
        
        return scores
    
    def execute(self, x, proposal_scores=None):
        """
        前向传播
        Args:
            x: RoI特征 [B, C, H, W]
            proposal_scores: 提议分数 [B] (可选)
        Returns:
            cls_scores: 分类分数 [B, num_classes]
            bbox_preds: 边界框预测 [B, num_classes*4]
        """
        # RoI特征提取
        x = self.avg_pool(x)
        
        # 展平特征
        x = x.view(x.shape[0], -1)
        
        # 特征投影
        x = self.fc(x)
        x = self.relu(x)
        
        # 分类和回归
        cls_scores = self.cls_head(x)
        bbox_preds = self.reg_head(x)
        
        # CLIP特征匹配
        if self.clip_embeddings is not None:
            # 归一化特征
            x_norm = jt.normalize(x, p=2, dim=1)
            clip_norm = jt.normalize(self.clip_embeddings, p=2, dim=1)
            
            # 计算余弦相似度
            clip_scores = jt.matmul(x_norm, clip_norm.t()) / self.temperature
            
            # 融合CLIP分数
            cls_scores = cls_scores + clip_scores * 0.5
        
        # 应用sigmoid激活
        cls_scores = jt.sigmoid(cls_scores)
        
        # 概率校准
        cls_scores = self.calibrate_scores(cls_scores, proposal_scores)
        
        return cls_scores, bbox_preds
    
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, 
                   scale_factor, rescale=False, cfg=None):
        """
        获取检测框 - 基于UniDetector的get_bboxes实现
        """
        # 分离分类分数和提议分数
        if isinstance(cls_score, (list, tuple)):
            cls_score, proposal_score = cls_score[0], cls_score[1]
        else:
            proposal_score = None
        
        # 应用sigmoid
        scores = jt.sigmoid(cls_score)
        
        # 概率校准
        if self.with_calibration:
            scores = self.calibrate_scores(scores, proposal_score)
        
        # 边界框解码
        if bbox_pred is not None:
            # 这里需要实现bbox_coder.decode
            # 简化实现：直接使用rois
            bboxes = rois[:, 1:].clone()
        else:
            bboxes = rois[:, 1:].clone()
        
        # 裁剪到图像边界
        if img_shape is not None:
            bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
            bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        
        # 缩放回原始图像尺寸
        if rescale and bboxes.shape[0] > 0:
            scale_factor = jt.array(scale_factor)
            bboxes = (bboxes.view(bboxes.shape[0], -1, 4) / scale_factor).view(
                bboxes.shape[0], -1)
        
        if cfg is None:
            return bboxes, scores
        else:
            # 这里需要实现multiclass_nms
            # 简化实现：直接返回
            return bboxes, scores 