import jittor as jt
import jittor.nn as nn
import numpy as np
import sys
import os

# 注册到 MMDet 的 Registry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import HEADS


def _l2_normalize_feature(x: jt.Var, dim: int = 1, eps: float = 1e-6) -> jt.Var:
    denom = jt.sqrt(jt.sum(x * x, dim=dim, keepdims=True) + eps)
    return x / denom


@HEADS.register_module(force=True)
class OlnRPNHead(nn.Module):
    """Jittor 版 OLN-RPN Head（简化）：
    - 支持 FPN 多层输入
    - 输出分类分支、回归分支、objectness 分支
    - 提供 get_bboxes（简化）以根据 objectness 生成 proposals（无需依赖外部训练脚本）
    注意：本实现未接入 assigner/sampler 与正式 bbox_coder，便于先跑通整体流程。
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        num_anchors: int = 1,
        anchor_generator: dict = None,
        loss_cls=None,
        loss_bbox=None,
        loss_objectness=None,
        objectness_type: str = 'Centerness',
        train_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.objectness_type = objectness_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 解析 anchors 与 strides（仅用于 get_bboxes 简化生成）
        self.anchor_generator_cfg = anchor_generator or {}
        self.anchor_scales = self.anchor_generator_cfg.get('scales', [8])
        self.anchor_ratios = self.anchor_generator_cfg.get('ratios', [1.0])
        self.strides = self.anchor_generator_cfg.get('strides', [4, 8, 16, 32, 64])
        # OLN 论文使用单 anchor/像素
        self.num_anchors = max(1, len(self.anchor_scales) * len(self.anchor_ratios))

        # 层
        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(feat_channels, self.num_anchors * 1, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, self.num_anchors * 4, 1)
        self.rpn_obj = nn.Conv2d(feat_channels, self.num_anchors, 1)

        self._init_weights()

    def _init_weights(self):
        jt.init.gauss_(self.rpn_conv.weight, std=0.01)
        jt.init.constant_(self.rpn_conv.bias, 0)
        jt.init.gauss_(self.rpn_cls.weight, std=0.01)
        jt.init.constant_(self.rpn_cls.bias, 0)
        jt.init.gauss_(self.rpn_reg.weight, std=0.01)
        jt.init.constant_(self.rpn_reg.bias, 0)
        jt.init.gauss_(self.rpn_obj.weight, std=0.01)
        jt.init.constant_(self.rpn_obj.bias, 0)

    def _forward_single(self, feat: jt.Var):
        x = self.rpn_conv(feat)
        x = nn.ReLU()(x)
        x = _l2_normalize_feature(x, dim=1)
        cls_score = self.rpn_cls(x)
        bbox_pred = self.rpn_reg(x)
        objectness = self.rpn_obj(x)
        return cls_score, bbox_pred, objectness

    def execute(self, feats):
        # feats: Var[B,C,H,W] or List[Var]
        if isinstance(feats, (list, tuple)):
            cls_scores, bbox_preds, objectness_scores = [], [], []
            for f in feats:
                cs, bp, ob = self._forward_single(f)
                cls_scores.append(cs)
                bbox_preds.append(bp)
                objectness_scores.append(ob)
            return cls_scores, bbox_preds, objectness_scores
        else:
            return self._forward_single(feats)

    def _nms_numpy(self, boxes_np, scores_np, iou_thr=0.7, max_num=1000):
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores_np.argsort()[::-1]
        keep = []
        while order.size > 0 and len(keep) < max_num:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def get_bboxes(self, cls_scores, bbox_preds, objectness_scores, img_shape, cfg=None):
        """基于 objectness_scores 生成 proposals（简化版）。
        - cls_scores, bbox_preds, objectness_scores: List[Var[B, A*k, H, W]]
        - img_shape: Var shape of input image (B,C,H,W)
        - cfg: 测试配置（含 nms_pre, max_num, nms_thr）
        返回: Var[N, 5]，列为 (b, x1, y1, x2, y2)
        """
        # 允许 cfg 为 None 或缺字段
        nms_pre = 1000
        max_num = 1000
        nms_thr = 0.7
        try:
            if cfg is not None:
                nms_pre = int(getattr(cfg, 'nms_pre', nms_pre))
                max_num = int(getattr(cfg, 'max_num', max_num))
                nms_thr = float(getattr(cfg, 'nms_thr', nms_thr))
        except Exception:
            pass

        if not isinstance(objectness_scores, (list, tuple)):
            objectness_scores = [objectness_scores]
        B = int(img_shape[0]) if hasattr(img_shape, '__len__') else 1

        rois_concat = []
        for b in range(B):
            boxes_all = []
            scores_all = []
            for lvl, obj_map in enumerate(objectness_scores):
                stride = self.strides[lvl] if lvl < len(self.strides) else self.strides[-1]
                H, W = int(obj_map.shape[2]), int(obj_map.shape[3])
                A = int(obj_map.shape[1])
                if H <= 0 or W <= 0 or A <= 0:
                    continue
                # objectness 概率
                probs = jt.sigmoid(obj_map[b:b+1, :, :, :]).reshape(-1)
                # 对应的 bbox 预测（TBLR 距离），按 stride 放缩到像素坐标
                if isinstance(bbox_preds, (list, tuple)) and len(bbox_preds) > lvl:
                    bp = bbox_preds[lvl][b:b+1, :, :, :]  # [1, A*4, H, W]
                    bp = bp.reshape(1, A, 4, H, W)
                    l = nn.ReLU()(bp[:, :, 0, :, :]) * stride
                    t = nn.ReLU()(bp[:, :, 1, :, :]) * stride
                    r = nn.ReLU()(bp[:, :, 2, :, :]) * stride
                    btm = nn.ReLU()(bp[:, :, 3, :, :]) * stride
                else:
                    # 缺省：使用固定尺度中心框
                    scale = float(self.anchor_scales[0]) if len(self.anchor_scales) > 0 else 8.0
                    half = (stride * scale)
                    l = jt.full((1, A, H, W), half)
                    t = jt.full((1, A, H, W), half)
                    r = jt.full((1, A, H, W), half)
                    btm = jt.full((1, A, H, W), half)
                # 网格中心 broadcast 到 (1, A, H, W)
                yy, xx = jt.meshgrid([jt.arange(H), jt.arange(W)])
                xx = (xx + 0.5) * stride
                yy = (yy + 0.5) * stride
                xx = xx.reshape(1, 1, H, W).broadcast((1, A, H, W))
                yy = yy.reshape(1, 1, H, W).broadcast((1, A, H, W))
                x1 = (xx - l).clamp(0, float(img_shape[3]-1))
                y1 = (yy - t).clamp(0, float(img_shape[2]-1))
                x2 = (xx + r).clamp(0, float(img_shape[3]-1))
                y2 = (yy + btm).clamp(0, float(img_shape[2]-1))
                # 展平到 (N,4)
                boxes = jt.stack([
                    x1.reshape(-1), y1.reshape(-1), x2.reshape(-1), y2.reshape(-1)
                ], dim=1)

                # 预筛选 top-k
                k = int(min(nms_pre, boxes.shape[0]))
                scores_np = probs.numpy()
                order = np.argsort(-scores_np)[:k]
                boxes_np = boxes.numpy()[order]
                scores_np = scores_np[order]
                # NMS
                keep = self._nms_numpy(boxes_np, scores_np, iou_thr=nms_thr, max_num=max_num)
                boxes_np = boxes_np[keep]
                scores_np = scores_np[keep]
                boxes_all.append(boxes_np)
                scores_all.append(scores_np)
            if len(boxes_all) == 0:
                continue
            boxes_all = np.concatenate(boxes_all, axis=0)
            scores_all = np.concatenate(scores_all, axis=0)
            order = np.argsort(-scores_all)[:max_num]
            boxes_all = boxes_all[order]
            b_col = np.full((boxes_all.shape[0], 1), float(b), dtype=np.float32)
            rois_b = np.concatenate([b_col, boxes_all.astype(np.float32)], axis=1)
            rois_concat.append(rois_b)
        if len(rois_concat) == 0:
            return jt.zeros((0, 5), dtype='float32')
        rois_np = np.concatenate(rois_concat, axis=0)
        return jt.array(rois_np)

    # ===== 训练所需：目标构建与损失 =====
    def _bbox_iou(self, boxes1, boxes2):
        # boxes: [N,4] x1,y1,x2,y2
        x1 = jt.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = jt.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = jt.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = jt.minimum(boxes1[:, 3], boxes2[:, 3])
        inter_w = jt.maximum(0.0, x2 - x1)
        inter_h = jt.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area1 = jt.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * jt.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
        area2 = jt.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * jt.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _build_level_targets(self, H, W, stride, gt_bboxes_b):
        # 生成网格中心
        yy, xx = jt.meshgrid([jt.arange(H), jt.arange(W)])
        xx = (xx + 0.5) * stride
        yy = (yy + 0.5) * stride
        xc = xx.reshape(-1)  # [HW]
        yc = yy.reshape(-1)
        num_pts = xc.shape[0]
        if gt_bboxes_b is None or gt_bboxes_b.shape[0] == 0:
            pos_mask = jt.zeros((num_pts,), dtype='bool')
            tblr_targets = jt.zeros((num_pts, 4), dtype='float32')
            centerness = jt.zeros((num_pts,), dtype='float32')
            matched = jt.full((num_pts,), -1, dtype='int32')
            return pos_mask, tblr_targets, centerness, matched, xc, yc
        # gt: [G,4]
        G = int(gt_bboxes_b.shape[0])
        x1 = gt_bboxes_b[:, 0].reshape(1, G)
        y1 = gt_bboxes_b[:, 1].reshape(1, G)
        x2 = gt_bboxes_b[:, 2].reshape(1, G)
        y2 = gt_bboxes_b[:, 3].reshape(1, G)
        # broadcast to [HW,G]
        xc_b = xc.reshape(-1, 1)
        yc_b = yc.reshape(-1, 1)
        l = xc_b - x1
        t = yc_b - y1
        r = x2 - xc_b
        btm = y2 - yc_b
        inside = (l > 0) & (t > 0) & (r > 0) & (btm > 0)
        # 选匹配 gt：采用最小 gt 面积
        areas = (x2 - x1) * (y2 - y1)  # [1,G]
        areas_b = jt.where(inside, areas, jt.full_like(areas, float('inf')))
        min_area, min_idx = jt.argmin(areas_b, dim=1)  # [HW]
        pos_mask = jt.isfinite(min_area)
        pos_mask = pos_mask & (min_idx >= 0)
        # 生成目标 tblr 与 centerness
        idx = jt.maximum(min_idx, jt.int32(0))
        # gather
        gather_idx = idx.reshape(-1, 1).repeat(1, 4)
        l_all = (xc_b - x1).gather(1, idx.reshape(-1, 1)).reshape(-1)
        t_all = (yc_b - y1).gather(1, idx.reshape(-1, 1)).reshape(-1)
        r_all = (x2 - xc_b).gather(1, idx.reshape(-1, 1)).reshape(-1)
        b_all = (y2 - yc_b).gather(1, idx.reshape(-1, 1)).reshape(-1)
        tblr = jt.stack([l_all, t_all, r_all, b_all], dim=1)
        tblr = jt.maximum(tblr, 0.0)
        # centerness
        lr_min = jt.minimum(l_all, r_all)
        lr_max = jt.maximum(l_all, r_all) + 1e-12
        tb_min = jt.minimum(t_all, b_all)
        tb_max = jt.maximum(t_all, b_all) + 1e-12
        centerness = jt.sqrt((lr_min / lr_max) * (tb_min / tb_max))
        return pos_mask, tblr, centerness, idx, xc, yc

    def _coerce_gt_boxes(self, gt):
        """将多样式 gt 输入转换为 jt.Var[N,4] float32。"""
        if gt is None:
            return jt.zeros((0, 4), dtype='float32')
        # 已是 jt.Var
        try:
            if isinstance(gt, jt.Var):
                if len(gt.shape) == 2 and gt.shape[1] == 4:
                    return gt.float32()
                return gt.reshape(-1, 4).float32()
        except Exception:
            pass
        # numpy / torch tensor
        if hasattr(gt, 'shape') and hasattr(gt, 'dtype'):
            try:
                return jt.array(gt).reshape(-1, 4).float32()
            except Exception:
                pass
        # 列表/嵌套列表
        if isinstance(gt, (list, tuple)):
            if len(gt) == 0:
                return jt.zeros((0, 4), dtype='float32')
            elems = []
            for item in gt:
                if item is None:
                    continue
                if isinstance(item, jt.Var):
                    if len(item.shape) == 2 and item.shape[1] == 4:
                        elems.append(item)
                    else:
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
                # 逐个 append 兼容形状
                out = elems[0]
                for e in elems[1:]:
                    try:
                        out = jt.concat([out, e], dim=0)
                    except Exception:
                        pass
                return out.float32()
        # 其他类型，返回空
        return jt.zeros((0, 4), dtype='float32')

    def loss(self, cls_scores, bbox_preds, objectness_scores, gt_bboxes_list, img_shape, pos_iou_thr=0.5):
        # 多层、多 batch
        if not isinstance(bbox_preds, (list, tuple)):
            bbox_preds = [bbox_preds]
        if not isinstance(objectness_scores, (list, tuple)):
            objectness_scores = [objectness_scores]
        B = int(img_shape[0]) if hasattr(img_shape, '__len__') else 1
        # 阈值
        pos_thr = None
        neg_thr = None
        try:
            if self.train_cfg and hasattr(self.train_cfg, 'objectness_assigner'):
                oa = self.train_cfg.objectness_assigner
                pos_thr = getattr(oa, 'pos_iou_thr', None)
                neg_thr = getattr(oa, 'neg_iou_thr', None)
        except Exception:
            pass
        if pos_thr is None:
            pos_thr = 0.3
        if neg_thr is None:
            neg_thr = 0.1
        # stride 与 scale
        strides = self.strides if isinstance(self.strides, (list, tuple)) else [4, 8, 16, 32]
        base_scale = float(self.anchor_scales[0]) if len(self.anchor_scales) > 0 else 8.0
        # 累积
        sum_reg = jt.zeros(1)
        sum_obj = jt.zeros(1)
        num_pos_total = 0
        num_obj_total = 0
        for b in range(B):
            gt_b = None
            if isinstance(gt_bboxes_list, (list, tuple)) and len(gt_bboxes_list) > b:
                gt_b = self._coerce_gt_boxes(gt_bboxes_list[b])
            if gt_b is None:
                gt_b = jt.zeros((0, 4), dtype='float32')
            for lvl, (bp, ob) in enumerate(zip(bbox_preds, objectness_scores)):
                stride = strides[lvl] if lvl < len(strides) else strides[-1]
                H, W = int(bp.shape[2]), int(bp.shape[3])
                if H <= 0 or W <= 0:
                    continue
                # 中心网格 (broadcast 到 A)
                A = max(1, int(ob.shape[1]))
                yy, xx = jt.meshgrid([jt.arange(H), jt.arange(W)])
                cx = (xx + 0.5) * stride
                cy = (yy + 0.5) * stride
                cx = cx.reshape(1, 1, H, W).broadcast((1, A, H, W))
                cy = cy.reshape(1, 1, H, W).broadcast((1, A, H, W))
                # 默认方框用于 IoU 匹配
                half = (stride * base_scale) / 2.0
                x1_a = (cx - half).reshape(-1)
                y1_a = (cy - half).reshape(-1)
                x2_a = (cx + half).reshape(-1)
                y2_a = (cy + half).reshape(-1)
                anchors_boxes = jt.stack([x1_a, y1_a, x2_a, y2_a], dim=1)  # [N,4]
                N = anchors_boxes.shape[0]
                if gt_b.shape[0] == 0:
                    # 仅计算 objectness 的负样本
                    probs = jt.sigmoid(ob[b:b+1, :, :, :].reshape(-1))
                    sum_obj += jt.abs(probs - 0.0).mean()
                    num_obj_total += N
                    continue
                # IoU 匹配，选最大 IoU GT
                # 展开 GT 为 [N,G,4]
                G = int(gt_b.shape[0])
                x1g = gt_b[:, 0].reshape(1, G).broadcast((N, G))
                y1g = gt_b[:, 1].reshape(1, G).broadcast((N, G))
                x2g = gt_b[:, 2].reshape(1, G).broadcast((N, G))
                y2g = gt_b[:, 3].reshape(1, G).broadcast((N, G))
                x1 = anchors_boxes[:, 0].reshape(N, 1).broadcast((N, G))
                y1 = anchors_boxes[:, 1].reshape(N, 1).broadcast((N, G))
                x2 = anchors_boxes[:, 2].reshape(N, 1).broadcast((N, G))
                y2 = anchors_boxes[:, 3].reshape(N, 1).broadcast((N, G))
                xx1 = jt.maximum(x1, x1g)
                yy1 = jt.maximum(y1, y1g)
                xx2 = jt.minimum(x2, x2g)
                yy2 = jt.minimum(y2, y2g)
                iw = jt.maximum(0.0, xx2 - xx1)
                ih = jt.maximum(0.0, yy2 - yy1)
                inter = iw * ih
                area_a = jt.maximum(0.0, x2 - x1) * jt.maximum(0.0, y2 - y1)
                area_g = jt.maximum(0.0, x2g - x1g) * jt.maximum(0.0, y2g - y1g)
                union = area_a + area_g - inter + 1e-6
                ious = inter / union  # [N,G]
                iou_max, gt_idx = jt.argmax(ious, dim=1)  # [N]
                pos_mask = (iou_max >= pos_thr)
                neg_mask = (iou_max <= neg_thr)
                # 预测解码为 boxes
                bp_b = bp[b:b+1, :, :, :].reshape(1, A, 4, H, W)
                lp = nn.ReLU()(bp_b[:, :, 0, :, :]).reshape(-1)
                tp = nn.ReLU()(bp_b[:, :, 1, :, :]).reshape(-1)
                rp = nn.ReLU()(bp_b[:, :, 2, :, :]).reshape(-1)
                bpv = nn.ReLU()(bp_b[:, :, 3, :, :]).reshape(-1)
                cxv = cx.reshape(-1)
                cyv = cy.reshape(-1)
                x1_p = (cxv - lp).clamp(0, float(img_shape[3]-1))
                y1_p = (cyv - tp).clamp(0, float(img_shape[2]-1))
                x2_p = (cxv + rp).clamp(0, float(img_shape[3]-1))
                y2_p = (cyv + bpv).clamp(0, float(img_shape[2]-1))
                boxes_p = jt.stack([x1_p, y1_p, x2_p, y2_p], dim=1)
                # GT gather
                gt_sel = gt_b[jt.maximum(gt_idx, jt.int32(0)), :]
                # 仅正样本参与回归
                if int(pos_mask.sum()) > 0:
                    pos_inds = (pos_mask.nonzero()).reshape(-1)
                    boxes_pos = boxes_p[pos_inds, :]
                    gt_pos = gt_sel[pos_inds, :]
                    iou_pos = self._bbox_iou(boxes_pos, gt_pos)
                    sum_reg += (1.0 - iou_pos).mean()
                    num_pos_total += 1
                # objectness 目标 centerness（对正样本用，负样本为 0，忽略中间区域）
                # 计算正样本的 TBLR，用于 centerness
                if int((pos_mask | neg_mask).sum()) > 0:
                    # 对所有点，构造目标值，mask 后聚合
                    # 对 pos 处的 TBLR 计算 centerness
                    x1g_sel = gt_sel[:, 0]
                    y1g_sel = gt_sel[:, 1]
                    x2g_sel = gt_sel[:, 2]
                    y2g_sel = gt_sel[:, 3]
                    l_t = (cxv - x1g_sel).clamp(0)
                    t_t = (cyv - y1g_sel).clamp(0)
                    r_t = (x2g_sel - cxv).clamp(0)
                    b_t = (y2g_sel - cyv).clamp(0)
                    lr_min = jt.minimum(l_t, r_t)
                    lr_max = jt.maximum(l_t, r_t) + 1e-12
                    tb_min = jt.minimum(t_t, b_t)
                    tb_max = jt.maximum(t_t, b_t) + 1e-12
                    centerness = jt.sqrt((lr_min / lr_max) * (tb_min / tb_max))
                    probs = jt.sigmoid(ob[b:b+1, :, :, :].reshape(-1))
                    mask_all = (pos_mask | neg_mask)
                    tgt = jt.zeros_like(centerness)
                    tgt = jt.where(pos_mask, centerness, tgt)
                    # L1 on selected indices
                    sel_inds = (mask_all.nonzero()).reshape(-1)
                    if int(sel_inds.shape[0]) > 0:
                        sum_obj += jt.abs(probs[sel_inds] - tgt[sel_inds]).mean()
                        num_obj_total += 1
        if num_pos_total == 0 and num_obj_total == 0:
            return {'loss_rpn_cls': jt.zeros(1), 'loss_rpn_bbox': jt.zeros(1), 'loss_rpn_obj': jt.zeros(1)}
        loss_bbox = sum_reg if num_pos_total > 0 else jt.zeros(1)
        loss_obj = sum_obj if num_obj_total > 0 else jt.zeros(1)
        return {'loss_rpn_cls': jt.zeros(1), 'loss_rpn_bbox': loss_bbox, 'loss_rpn_obj': loss_obj}


