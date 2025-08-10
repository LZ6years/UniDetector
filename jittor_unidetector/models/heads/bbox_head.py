import jittor as jt
import jittor.nn as nn
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS

@HEADS.register_module(force=True)
class BBoxHead(nn.Module):
    """Jittor版本的BBoxHead基类"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 **kwargs):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.roi_feat_area = roi_feat_size * roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic

        # 构建损失函数
        self.loss_cls = self._build_loss(loss_cls)
        self.loss_bbox = self._build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            in_channels *= self.roi_feat_area
            
        if self.with_cls:
            cls_channels = num_classes + 1  # 包含背景类
            self.fc_cls = nn.Linear(in_channels, cls_channels)
            
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)

    def _build_loss(self, loss_cfg):
        """构建损失函数"""
        loss_type = loss_cfg.get('type', 'CrossEntropyLoss')
        loss_weight = loss_cfg.get('loss_weight', 1.0)
        
        if loss_type == 'CrossEntropyLoss':
            use_sigmoid = loss_cfg.get('use_sigmoid', False)
            loss = nn.CrossEntropyLoss()
        elif loss_type == 'SmoothL1Loss':
            beta = loss_cfg.get('beta', 1.0)
            loss = nn.SmoothL1Loss(beta=beta)
        elif loss_type == 'L1Loss':
            loss = nn.L1Loss()
        elif loss_type == 'MSELoss':
            loss = nn.MSELoss()
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
            cls_score: 分类分数 [B, num_classes+1]
            bbox_pred: 边界框预测 [B, 4]
        """
        if self.with_avg_pool:
            x = self.avg_pool(x)
        
        x = x.view(x.shape[0], -1)
        
        cls_score = None
        bbox_pred = None
        
        if self.with_cls:
            cls_score = self.fc_cls(x)
            
        if self.with_reg:
            bbox_pred = self.fc_reg(x)
            
        return cls_score, bbox_pred

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        """
        计算损失
        Args:
            cls_score: 分类分数
            bbox_pred: 边界框预测
            rois: RoI区域
            labels: 标签
            label_weights: 标签权重
            bbox_targets: 边界框目标
            bbox_weights: 边界框权重
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        if cls_score is not None:
            # 分类损失
            # 确保label_weights是Jittor张量
            if not isinstance(label_weights, jt.Var):
                label_weights = jt.array(label_weights)
            avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(cls_score, labels, weight=label_weights, avg_factor=avg_factor)
            
        if bbox_pred is not None:
            # 回归损失
            pos_inds = labels > 0
            if pos_inds.sum() > 0:
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred, pos_bbox_targets, weight=pos_bbox_weights)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                
        return losses

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        """
        获取边界框
        Args:
            rois: RoI区域
            cls_score: 分类分数
            bbox_pred: 边界框预测
            img_shape: 图像形状
            scale_factor: 缩放因子
            rescale: 是否重新缩放
            cfg: 配置
        Returns:
            det_bboxes: 检测到的边界框
            det_labels: 检测到的标签
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
            bboxes /= bboxes.new_tensor(scale_factor)
            
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = self.multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    def multiclass_nms(self, multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
        """多类别NMS"""
        # 简化实现，实际应该使用更完整的NMS
        if multi_scores is None:
            return multi_bboxes, None
            
        scores = multi_scores[:, :-1]  # 排除背景类
        labels = jt.arange(self.num_classes, dtype=jt.long)
        labels = labels.view(1, -1).expand_as(scores)
        
        # 过滤低分数
        valid_mask = scores > score_thr
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]
        valid_bboxes = multi_bboxes[valid_mask.view(-1)]
        
        return valid_bboxes, valid_labels

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                jt.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
