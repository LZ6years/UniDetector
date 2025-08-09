import jittor as jt
import jittor.nn as nn
import numpy as np
import clip
import os
import sys

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS

@HEADS.register_module(force=True)
class CLIPBBoxHead(nn.Module):
    """
    CLIP边界框头
    使用CLIP特征进行零样本分类，支持概率校准
    """
    
    def __init__(self, in_channels, roi_feat_size=7, num_classes=80, 
                 clip_embedding_path=None, temperature=0.07,
                 with_calibration=True, beta=0.3, gamma=0.6,
                 prior_prob_path=None, **kwargs):
        super(CLIPBBoxHead, self).__init__()
        
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.with_calibration = with_calibration
        self.beta = beta
        self.gamma = gamma
        
        # RoI特征提取
        self.avg_pool = nn.AdaptiveAvgPool2d((roi_feat_size, roi_feat_size))
        
        # 特征投影层
        self.fc = nn.Linear(in_channels * roi_feat_size * roi_feat_size, 1024)
        self.relu = nn.ReLU()
        
        # 加载CLIP文本嵌入
        self.clip_embeddings = self.load_clip_embeddings(clip_embedding_path)
        
        # 加载先验概率
        self.prior_probs = None
        if prior_prob_path and os.path.exists(prior_prob_path):
            self.prior_probs = np.load(prior_prob_path)
            print(f"Loaded prior probabilities from {prior_prob_path}")
        
        # 初始化权重
        self.init_weights()
    
    def load_clip_embeddings(self, embedding_path):
        """
        加载CLIP文本嵌入
        Args:
            embedding_path: 嵌入文件路径
        Returns:
            embeddings: CLIP文本嵌入 [num_classes, 512]
        """
        if embedding_path and jt.misc.exists(embedding_path):
            embeddings = np.load(embedding_path)
        else:
            # 如果没有预计算的嵌入，使用随机初始化
            embeddings = np.random.randn(self.num_classes, 512).astype(np.float32)
        
        return jt.array(embeddings)
    
    def init_weights(self):
        """初始化权重"""
        jt.init.gauss_(self.fc.weight, std=0.01)
        jt.init.constant_(self.fc.bias, 0)
    
    def calibrate_scores(self, scores, proposal_scores=None):
        """
        概率校准 - 基于UniDetector原始实现
        使用公式: scores = scores^beta * proposal_score^(1-beta)
        频率校准: scores = scores * (1/frequencies^gamma) / mean(frequencies)
        Args:
            scores: 原始分类分数 [B, num_classes]
            proposal_scores: 提议分数 [B]
        Returns:
            calibrated_scores: 校准后的分数
        """
        if not self.with_calibration:
            return scores
        
        # 1. 频率校准 (基于检测结果统计)
        if hasattr(self, 'cnum') and self.cnum is not None:
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
    
    def execute(self, x):
        """
        前向传播
        Args:
            x: RoI特征 [B, C, H, W]
        Returns:
            cls_scores: 分类分数 [B, num_classes]
            bbox_preds: 边界框预测 [B, num_classes*4]
        """
        # RoI特征提取
        x = self.avg_pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 特征投影
        x = self.fc(x)
        x = self.relu(x)
        
        # 与CLIP嵌入计算相似度
        # 归一化特征
        x_norm = jt.normalize(x, p=2, dim=1)
        clip_norm = jt.normalize(self.clip_embeddings, p=2, dim=1)
        
        # 计算余弦相似度
        cls_scores = jt.matmul(x_norm, clip_norm.t()) / self.temperature
        
        # 概率校准
        cls_scores = self.calibrate_scores(cls_scores)
        
        # 边界框回归（简化版本）
        bbox_preds = jt.zeros((x.size(0), self.num_classes * 4))
        
        return cls_scores, bbox_preds


@HEADS.register_module(force=True)
class CLIPFeatureExtractor(nn.Module):
    """
    CLIP特征提取器
    用于提取图像和文本特征
    """
    
    def __init__(self, clip_model_name="RN50", **kwargs):
        super(CLIPFeatureExtractor, self).__init__()
        
        # 加载CLIP模型
        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        
    def extract_image_features(self, images):
        """
        提取图像特征
        Args:
            images: 图像张量 [B, 3, H, W]
        Returns:
            features: 图像特征 [B, 512]
        """
        features = self.clip_model.encode_image(images)
        return features
    
    def extract_text_features(self, text):
        """
        提取文本特征
        Args:
            text: 文本列表
        Returns:
            features: 文本特征 [N, 512]
        """
        # 对文本进行编码
        text_tokens = clip.tokenize(text)
        features = self.clip_model.encode_text(text_tokens)
        return features
    
    def compute_similarity(self, image_features, text_features):
        """
        计算图像和文本特征的相似度
        Args:
            image_features: 图像特征 [B, 512]
            text_features: 文本特征 [N, 512]
        Returns:
            similarity: 相似度矩阵 [B, N]
        """
        # 归一化特征
        image_features = jt.normalize(image_features, p=2, dim=1)
        text_features = jt.normalize(text_features, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = jt.matmul(image_features, text_features.t())
        
        return similarity 