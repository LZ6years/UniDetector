import jittor as jt
import jittor.nn as nn
import numpy as np
import pickle
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS
from ...bbox_head import BBoxHead

@HEADS.register_module(force=True)
class BBoxHeadCLIPInference(BBoxHead):
    """
    CLIP边界框推理头，支持概率校准
    参考UniDetector的BBoxHeadCLIPInference实现
    """
    
    def __init__(self,
                 with_avg_pool=True,
                 roi_feat_size=7,
                 in_channels=2048,
                 num_classes=1203,
                 zeroshot_path=None,
                 withcalibration=False,
                 resultfile=None,
                 gamma=0.3,
                 beta=0.8,
                 **kwargs):
        super(BBoxHeadCLIPInference, self).__init__(
            with_avg_pool=with_avg_pool,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            **kwargs)
        
        self.num_classes = num_classes
        self.withcalibration = withcalibration
        self.resultfile = resultfile
        self.gamma = gamma
        self.beta = beta
        
        # 加载CLIP嵌入
        if zeroshot_path is not None:
            self.zs_weight = self._load_clip_embeddings(zeroshot_path)
        else:
            self.zs_weight = None
        
        # 加载类别频率（用于概率校准）
        if self.withcalibration and self.resultfile is not None:
            self.cnum = self._load_class_frequencies()
        else:
            self.cnum = None
    
    def _load_clip_embeddings(self, zeroshot_path):
        """加载CLIP嵌入"""
        try:
            embeddings = np.load(zeroshot_path)
            zs_weight = jt.array(embeddings, dtype='float32')
            print(f"Loaded CLIP embeddings from {zeroshot_path}, shape: {zs_weight.shape}")
            return zs_weight
        except Exception as e:
            print(f"Failed to load CLIP embeddings from {zeroshot_path}: {e}")
            return None
    
    def _load_class_frequencies(self):
        """从原始结果文件中加载类别频率"""
        try:
            with open(self.resultfile, 'rb') as f:
                results = pickle.load(f)
            
            # 统计每个类别的检测数量
            class_counts = np.zeros(self.num_classes)
            for result in results:
                if 'bbox_results' in result:
                    bbox_results = result['bbox_results']
                    for class_id, bboxes in enumerate(bbox_results):
                        if len(bboxes) > 0:
                            class_counts[class_id] += len(bboxes)
            
            print(f"Loaded class frequencies, shape: {class_counts.shape}")
            return class_counts
        except Exception as e:
            print(f"Failed to load class frequencies from {self.resultfile}: {e}")
            return None
    
    def execute(self, x):
        """
        前向传播
        Args:
            x: RoI特征 [B, C, H, W]
        Returns:
            cls_score: 分类分数 [B, num_classes]
            bbox_pred: 边界框预测 [B, 4]
        """
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        
        # 使用CLIP嵌入进行零样本分类
        if self.zs_weight is not None:
            # 计算CLIP相似度
            x_norm = jt.normalize(x, p=2, dim=1)
            zs_weight_norm = jt.normalize(self.zs_weight, p=2, dim=1)
            cls_score = jt.matmul(x_norm, zs_weight_norm.t())
        else:
            # 使用传统的全连接层
            cls_score = self.fc_cls(x)
        
        # 边界框回归
        bbox_pred = self.fc_reg(x)
        
        return cls_score, bbox_pred
    
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        """获取边界框，包含概率校准"""
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        
        # 应用概率校准
        if self.withcalibration and self.cnum is not None:
            cls_score = self._apply_probability_calibration(cls_score)
        
        scores = jt.softmax(cls_score, dim=1) if cls_score is not None else None
        
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        
        if rescale and scale_factor is not None:
            bboxes /= scale_factor
        
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = self.multiclass_nms(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels
    
    def _apply_probability_calibration(self, cls_score):
        """应用概率校准"""
        if self.cnum is None:
            return cls_score
        
        # 将类别频率转换为张量
        frequencies = jt.array(self.cnum, dtype='float32').view(1, -1).to(cls_score.device)
        
        # 避免除零错误
        frequencies = 1 / (frequencies + 0.000001) ** self.gamma
        
        # 应用校准
        scores = jt.sigmoid(cls_score)
        scores[:, :-1] = scores[:, :-1] * frequencies / frequencies.mean()
        
        # 转换回logits
        calibrated_cls_score = jt.log(scores / (1 - scores + 1e-8))
        
        return calibrated_cls_score
    
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        """推理时不计算损失"""
        return dict() 