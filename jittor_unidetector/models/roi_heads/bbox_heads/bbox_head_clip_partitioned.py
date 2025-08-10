import jittor as jt
import jittor.nn as nn
import numpy as np
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS
from models.heads.bbox_head import BBoxHead


@HEADS.register_module(force=True)
class BBoxHeadCLIPPartitioned(BBoxHead):
    """
    分区CLIP边界框头，支持多数据集训练
    参考UniDetector的BBoxHeadCLIPPartitioned实现
    """
    
    def __init__(self,
                 with_avg_pool=True,
                 roi_feat_size=7,
                 in_channels=2048,
                 num_classes=365,
                 zeroshot_path=None,
                 cat_freq_path=None,
                 dataset_id=0,
                 **kwargs):
        super(BBoxHeadCLIPPartitioned, self).__init__(
            with_avg_pool=with_avg_pool,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            **kwargs)
        
        self.num_classes = num_classes
        self.dataset_id = dataset_id
        
        # 加载CLIP嵌入
        if zeroshot_path is not None:
            if isinstance(zeroshot_path, list):
                self.zeroshot_path = zeroshot_path[dataset_id]
            else:
                self.zeroshot_path = zeroshot_path
            self.zs_weight = self._load_clip_embeddings()
        else:
            self.zs_weight = None
        
        # 加载类别频率
        if cat_freq_path is not None:
            if isinstance(cat_freq_path, list):
                self.cat_freq_path = cat_freq_path[dataset_id]
            else:
                self.cat_freq_path = cat_freq_path
            self.cat_freq = self._load_cat_freq()
        else:
            self.cat_freq = None
    
    def _load_clip_embeddings(self):
        """加载CLIP嵌入"""
        if self.zeroshot_path is None:
            return None
        
        try:
            embeddings = np.load(self.zeroshot_path)
            # 转换为Jittor张量
            zs_weight = jt.array(embeddings, dtype='float32')
            print(f"Loaded CLIP embeddings from {self.zeroshot_path}, shape: {zs_weight.shape}")
            return zs_weight
        except Exception as e:
            print(f"Failed to load CLIP embeddings from {self.zeroshot_path}: {e}")
            return None
    
    def _load_cat_freq(self):
        """加载类别频率"""
        if self.cat_freq_path is None:
            return None
        
        try:
            import json
            with open(self.cat_freq_path, 'r') as f:
                cat_freq = json.load(f)
            print(f"Loaded category frequencies from {self.cat_freq_path}")
            return cat_freq
        except Exception as e:
            print(f"Failed to load category frequencies from {self.cat_freq_path}: {e}")
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
    
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        """计算损失"""
        losses = dict()
        
        # 分类损失
        if cls_score is not None:
            # 确保label_weights是Jittor张量
            if not isinstance(label_weights, jt.Var):
                label_weights = jt.array(label_weights)
            avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
            losses['acc'] = self.accuracy(cls_score, labels)
        
        # 回归损失
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.sum() > 0:
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], 4)[pos_inds]
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred, pos_bbox_targets, pos_bbox_weights, avg_factor=pos_bbox_targets.shape[0])
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        
        return losses
    
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        """获取边界框"""
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
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