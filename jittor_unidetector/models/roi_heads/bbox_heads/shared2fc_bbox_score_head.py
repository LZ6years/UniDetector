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
        
        # 共享的全连接层（输入维度需根据是否 avg_pool 区分）
        input_dim = (self.in_channels if with_avg_pool
                     else self.in_channels * self.roi_feat_size * self.roi_feat_size)
        self.shared_fcs = nn.Sequential(
            nn.Linear(input_dim, fc_out_channels),
            nn.ReLU(),
            nn.Linear(fc_out_channels, fc_out_channels),
            nn.ReLU()
        )
        
        # 分类层（包含背景类，输出通道 = num_classes + 1）
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
        
        # 展平（Jittor 需使用 shape 索引，避免将 list 传入 reshape）
        x = x.view(x.shape[0], -1)
        
        # 共享全连接层
        x = self.shared_fcs(x)
        
        # 分类分支
        cls_score = self.fc_cls(x) if self.with_cls else None
        
        # 回归分支
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        
        # 边界框评分分支
        bbox_score = self.fc_bbox_score(x) if self.with_bbox_score else None
        
        return cls_score, bbox_pred, bbox_score

    def _bbox_iou(self, boxes1, boxes2):
        x1 = jt.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = jt.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = jt.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = jt.minimum(boxes1[:, 3], boxes2[:, 3])
        inter = jt.maximum(0.0, x2 - x1) * jt.maximum(0.0, y2 - y1)
        area1 = jt.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * jt.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
        area2 = jt.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * jt.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _delta_xywh_encode(self, rois_xyxy, gt_xyxy):
        # rois/gt: [N,4] in xyxy
        px = (rois_xyxy[:, 0] + rois_xyxy[:, 2]) * 0.5
        py = (rois_xyxy[:, 1] + rois_xyxy[:, 3]) * 0.5
        pw = (rois_xyxy[:, 2] - rois_xyxy[:, 0]).clamp(1.0)
        ph = (rois_xyxy[:, 3] - rois_xyxy[:, 1]).clamp(1.0)
        gx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) * 0.5
        gy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) * 0.5
        gw = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(1.0)
        gh = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(1.0)
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = jt.log(gw / pw)
        dh = jt.log(gh / ph)
        return jt.stack([dx, dy, dw, dh], dim=1)

    def build_targets_minimal(self, rois, gt_bboxes_list, img_shape, pos_iou_thr=0.5, neg_iou_thr=0.5, num_samples=256, pos_fraction=0.25):
        """
        构建简化版 ROI 训练目标：IoU 匹配 + 随机采样 + Delta 编码 + IoU 评分。
        rois: [N,5] (b,x1,y1,x2,y2)
        gt_bboxes_list: List[Var[Gi,4]]
        返回: labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets, bbox_score_weights
        """
        if rois is None or rois.shape[0] == 0:
            Z = 0
            zeros = jt.zeros((0,), dtype='float32')
            zeros4 = jt.zeros((0, 4), dtype='float32')
            labels = jt.zeros((0,), dtype='int32')
            return labels, zeros, zeros4, zeros4, zeros, zeros

        num_imgs = int(jt.ceil(rois[:, 0].max() + 1).item()) if rois.shape[0] > 0 else 1
        labels_all = []
        label_w_all = []
        bbox_t_all = []
        bbox_w_all = []
        iou_t_all = []
        iou_w_all = []

        def _coerce_gt_boxes(gt):
            if gt is None:
                return jt.zeros((0, 4), dtype='float32')
            try:
                if isinstance(gt, jt.Var):
                    if len(gt.shape) == 2 and gt.shape[1] == 4:
                        return gt.float32()
                    return gt.reshape(-1, 4).float32()
            except Exception:
                pass
            if hasattr(gt, 'shape') and hasattr(gt, 'dtype'):
                try:
                    return jt.array(gt).reshape(-1, 4).float32()
                except Exception:
                    pass
            if isinstance(gt, (list, tuple)):
                if len(gt) == 0:
                    return jt.zeros((0, 4), dtype='float32')
                elems = []
                for item in gt:
                    if item is None:
                        continue
                    if isinstance(item, jt.Var):
                        elems.append(item.reshape(-1, 4))
                    elif hasattr(item, 'shape') and hasattr(item, 'dtype'):
                        try:
                            elems.append(jt.array(item).reshape(-1, 4))
                        except Exception:
                            continue
                    elif isinstance(item, (list, tuple)) and len(item) == 4:
                        elems.append(jt.array(item).reshape(-1, 4))
                if len(elems) == 0:
                    return jt.zeros((0, 4), dtype='float32')
                try:
                    return jt.concat(elems, dim=0).float32()
                except Exception:
                    out = elems[0]
                    for e in elems[1:]:
                        try:
                            out = jt.concat([out, e], dim=0)
                        except Exception:
                            pass
                    return out.float32()
            return jt.zeros((0, 4), dtype='float32')

        for b in range(num_imgs):
            nz = (rois[:, 0] == float(b)).nonzero()
            if nz.shape[0] == 0:
                continue
            idxs = nz.reshape(-1)
            rois_b = rois[idxs, 1:5]
            gt_b = None
            if isinstance(gt_bboxes_list, (list, tuple)) and len(gt_bboxes_list) > b:
                gt_b = _coerce_gt_boxes(gt_bboxes_list[b])
            if gt_b is None or gt_b.shape[0] == 0:
                # 全部为背景
                num = int(rois_b.shape[0])
                labels = jt.full((num,), self.num_classes, dtype='int32')
                label_w = jt.ones((num,), dtype='float32')
                bbox_t = jt.zeros((num, 4), dtype='float32')
                bbox_w = jt.zeros((num, 4), dtype='float32')
                iou_t = jt.zeros((num,), dtype='float32')
                iou_w = jt.ones((num,), dtype='float32')
            else:
                # IoU 匹配
                N = int(rois_b.shape[0])
                G = int(gt_b.shape[0])
                x1g = gt_b[:, 0].reshape(1, G).broadcast((N, G))
                y1g = gt_b[:, 1].reshape(1, G).broadcast((N, G))
                x2g = gt_b[:, 2].reshape(1, G).broadcast((N, G))
                y2g = gt_b[:, 3].reshape(1, G).broadcast((N, G))
                x1 = rois_b[:, 0].reshape(N, 1).broadcast((N, G))
                y1 = rois_b[:, 1].reshape(N, 1).broadcast((N, G))
                x2 = rois_b[:, 2].reshape(N, 1).broadcast((N, G))
                y2 = rois_b[:, 3].reshape(N, 1).broadcast((N, G))
                xx1 = jt.maximum(x1, x1g)
                yy1 = jt.maximum(y1, y1g)
                xx2 = jt.minimum(x2, x2g)
                yy2 = jt.minimum(y2, y2g)
                inter = jt.maximum(0.0, xx2 - xx1) * jt.maximum(0.0, yy2 - yy1)
                area_a = jt.maximum(0.0, x2 - x1) * jt.maximum(0.0, y2 - y1)
                area_g = jt.maximum(0.0, x2g - x1g) * jt.maximum(0.0, y2g - y1g)
                union = area_a + area_g - inter + 1e-6
                ious = inter / union
                iou_max, gt_idx = jt.argmax(ious, dim=1)
                pos_mask = (iou_max >= pos_iou_thr)
                neg_mask = (iou_max < neg_iou_thr)
                # 采样
                pos_inds = (pos_mask.nonzero()).reshape(-1)
                neg_inds = (neg_mask.nonzero()).reshape(-1)
                max_pos = int(num_samples * pos_fraction)
                if int(pos_inds.shape[0]) > max_pos:
                    pos_inds = pos_inds[:max_pos]
                num_neg_need = int(num_samples - int(pos_inds.shape[0]))
                if int(neg_inds.shape[0]) > num_neg_need:
                    neg_inds = neg_inds[:num_neg_need]
                keep = jt.concat([pos_inds, neg_inds], dim=0) if int(neg_inds.shape[0]) > 0 else pos_inds
                if int(keep.shape[0]) == 0:
                    keep = jt.arange(min(num_samples, N))
                rois_s = rois_b[keep, :]
                iou_s = iou_max[keep]
                gt_s = gt_b[jt.maximum(gt_idx[keep], jt.int32(0)), :]
                # 标签与权重
                labels = jt.full((int(keep.shape[0]),), self.num_classes, dtype='int32')
                labels[:int(pos_inds.shape[0])] = 0  # 前景索引 0
                label_w = jt.ones((int(keep.shape[0]),), dtype='float32')
                # 回归目标（数值稳定：宽高下限1像素）
                bbox_t = self._delta_xywh_encode(rois_s, gt_s)
                bbox_w = jt.zeros_like(bbox_t)
                if int(pos_inds.shape[0]) > 0:
                    bbox_w[:int(pos_inds.shape[0]), :] = 1.0
                # BBox 评分目标（IoU）
                iou_t = iou_s.clamp(0.0, 1.0)
                iou_w = jt.ones_like(iou_t)
            labels_all.append(labels)
            label_w_all.append(label_w)
            bbox_t_all.append(bbox_t)
            bbox_w_all.append(bbox_w)
            iou_t_all.append(iou_t)
            iou_w_all.append(iou_w)

        labels = jt.concat(labels_all, dim=0) if len(labels_all) else jt.zeros((0,), dtype='int32')
        label_weights = jt.concat(label_w_all, dim=0) if len(label_w_all) else jt.zeros((0,), dtype='float32')
        bbox_targets = jt.concat(bbox_t_all, dim=0) if len(bbox_t_all) else jt.zeros((0, 4), dtype='float32')
        bbox_weights = jt.concat(bbox_w_all, dim=0) if len(bbox_w_all) else jt.zeros((0, 4), dtype='float32')
        bbox_score_targets = jt.concat(iou_t_all, dim=0) if len(iou_t_all) else jt.zeros((0,), dtype='float32')
        bbox_score_weights = jt.concat(iou_w_all, dim=0) if len(iou_w_all) else jt.zeros((0,), dtype='float32')
        return labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets, bbox_score_weights
    
    def loss(self, cls_score, bbox_pred, bbox_score, rois, labels, label_weights, 
             bbox_targets, bbox_weights, bbox_score_targets=None, bbox_score_weights=None, 
             reduction_override=None):
        """
        计算损失
        """
        losses = dict()
        
        def _accuracy(pred, target):
            try:
                if pred is None or target is None or pred.shape[0] == 0:
                    return jt.zeros(1)
                pred_label = jt.argmax(pred, dim=1)
                return (pred_label == target).float32().mean()
            except Exception:
                return jt.zeros(1)
        
        # 分类损失
        if cls_score is not None:
            # 当权重为 0 或无样本时跳过分类损失，避免 0×NaN
            cls_w = float(getattr(self.loss_cls, 'loss_weight', 1.0)) if hasattr(self, 'loss_cls') else 1.0
            if cls_w <= 0.0 or labels is None or (hasattr(labels, 'shape') and labels.shape[0] == 0):
                losses['loss_cls'] = jt.zeros(1)
                losses['acc'] = jt.zeros(1)
            else:
                # 使用稳定的 log-softmax + NLL 组合，减少数值溢出
                try:
                    log_probs = nn.log_softmax(cls_score, dim=1)
                    nll = -log_probs[jt.arange(labels.shape[0]), labels]
                    losses['loss_cls'] = nll.mean() * cls_w
                except Exception:
                    losses['loss_cls'] = self.loss_cls(cls_score, labels) * cls_w
                losses['acc'] = _accuracy(cls_score, labels)
        
        # 回归损失
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.sum() > 0:
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], 4)[pos_inds]
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred, pos_bbox_targets) * getattr(self.loss_bbox, 'loss_weight', 1.0)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        
        # 边界框评分损失
        if bbox_score is not None and bbox_score_targets is not None:
            losses['loss_bbox_score'] = self.loss_bbox_score(bbox_score.squeeze(), bbox_score_targets) * getattr(self.loss_bbox_score, 'loss_weight', 1.0)
        
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