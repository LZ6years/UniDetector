import jittor as jt
import jittor.nn as nn
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS
from models.heads.bbox_head import BBoxHead

@HEADS.register_module(force=True)
class Shared2FCBBoxScoreHead(BBoxHead):
    """
    用Jittor重写的Shared2FCBBoxScoreHead
    用于OLN-Box的边界框评分头
    """
    
    def __init__(self,
                 with_avg_pool=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=1,  # 类无关
                 fc_out_channels=1024,
                 with_bbox_score=True,
                 bbox_score_type='BoxIoU',
                 loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
                 alpha=0.3,
                 **kwargs):
        super(Shared2FCBBoxScoreHead, self).__init__(
            with_avg_pool=with_avg_pool,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs)
        
        self.fc_out_channels = fc_out_channels
        self.with_bbox_score = with_bbox_score
        self.bbox_score_type = bbox_score_type
        self.alpha = alpha
        
        # 共享的全连接层
        self.shared_fcs = nn.Sequential(
            nn.Linear(self.in_channels * self.roi_feat_size * self.roi_feat_size, fc_out_channels),
            nn.ReLU(),
            nn.Linear(fc_out_channels, fc_out_channels),
            nn.ReLU()
        )
        
        # 分类层
        if self.with_cls:
            self.fc_cls = nn.Linear(fc_out_channels, num_classes)
        
        # 回归层
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(fc_out_channels, out_dim_reg)
        
        # 边界框评分层
        if self.with_bbox_score:
            self.fc_bbox_score = nn.Linear(fc_out_channels, 1)
        
        # 损失函数
        if self.with_bbox_score:
            self.loss_bbox_score = self._build_loss(loss_bbox_score)
        
        # 检查损失权重（从父类继承的损失函数可能没有loss_weight属性）
        self.with_class_score = getattr(self.loss_cls, 'loss_weight', 1.0) > 0.0 if hasattr(self, 'loss_cls') else False
        self.with_bbox_loc_score = getattr(self.loss_bbox_score, 'loss_weight', 1.0) > 0.0 if hasattr(self, 'loss_bbox_score') else False
    
    def _build_loss(self, loss_cfg):
        """构建损失函数"""
        loss_type = loss_cfg.get('type', 'L1Loss')
        loss_weight = loss_cfg.get('loss_weight', 1.0)
        
        if loss_type == 'L1Loss':
            loss = nn.L1Loss()
        elif loss_type == 'MSELoss':
            loss = nn.MSELoss()
        elif loss_type == 'CrossEntropyLoss':
            loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # 添加loss_weight属性
        loss.loss_weight = loss_weight
        return loss
    
    def execute(self, x):
        """
        前向传播
        Args:
            x: RoI特征 [B, C, H, W]
        Returns:
            cls_score: 分类分数 [B, num_classes]
            bbox_pred: 边界框预测 [B, 4]
            bbox_score: 边界框评分 [B, 1]
        """
        # 平均池化
        if self.with_avg_pool:
            x = self.avg_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 共享全连接层
        x = self.shared_fcs(x)
        
        # 分类分支
        cls_score = self.fc_cls(x) if self.with_cls else None
        
        # 回归分支
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        
        # 边界框评分分支
        bbox_score = self.fc_bbox_score(x) if self.with_bbox_score else None
        
        return cls_score, bbox_pred, bbox_score
    
    def loss(self, cls_score, bbox_pred, bbox_score, rois, labels, label_weights, 
             bbox_targets, bbox_weights, bbox_score_targets=None, bbox_score_weights=None, 
             reduction_override=None):
        """
        计算损失
        """
        losses = dict()
        
        # 分类损失
        if cls_score is not None:
            avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
            losses['acc'] = self.accuracy(cls_score, labels)
        
        # 回归损失
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.sum() > 0:
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)[pos_inds]
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred, pos_bbox_targets, pos_bbox_weights, avg_factor=pos_bbox_targets.size(0))
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        
        # 边界框评分损失
        if bbox_score is not None and bbox_score_targets is not None:
            if bbox_score_weights is not None:
                avg_factor = max(jt.sum(bbox_score_weights > 0).float().item(), 1.)
            else:
                avg_factor = bbox_score_targets.size(0)
            losses['loss_bbox_score'] = self.loss_bbox_score(
                bbox_score.squeeze(), bbox_score_targets, 
                weight=bbox_score_weights, avg_factor=avg_factor)
        
        return losses
    
    def get_bboxes(self, rois, cls_score, bbox_pred, bbox_score, img_shape, scale_factor, rescale=False, cfg=None):
        """
        获取边界框
        """
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
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                jt.init.gauss_(m.weight, std=0.01)
                jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0) 