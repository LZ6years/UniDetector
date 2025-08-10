import jittor as jt
import jittor.nn as nn
import math
import sys
import os

# 引入 MMDet 的注册表（桥接）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module(force=True)
class SingleRoIExtractor(nn.Module):
    """
    Jittor 版简化 SingleRoIExtractor：
    - 兼容 FPN 多层特征输入
    - 基于 bbox 尺寸选择金字塔层级
    - 对选中层级局部裁剪后，用 adaptive_avg_pool2d 近似 RoIAlign 到固定尺寸
    注意：这里未实现精确的亚像素对齐（sampling_ratio/bilinear），后续可替换为更精确实现。
    """

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56, **kwargs):
        super().__init__()
        self.roi_layer_cfg = roi_layer or dict(type='RoIAlign', output_size=7, sampling_ratio=0)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.output_size = self.roi_layer_cfg.get('output_size', 7)

    def _map_roi_levels(self, rois, num_levels):
        # rois: [N, 5] (batch_idx, x1, y1, x2, y2) in image coords
        areas = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        target_lvls = jt.floor(jt.log2(jt.sqrt(jt.clamp(areas, min_v=1e-6)) / self.finest_scale) + 4)
        target_lvls = target_lvls.clamp(0, num_levels - 1).int32()
        return target_lvls

    def _crop_and_resize(self, feat, boxes, stride):
        # feat: [B, C, H, W], boxes: [M, 5] (b, x1, y1, x2, y2) in image coords
        # 返回 [M, C, output_size, output_size]
        if boxes.shape[0] == 0:
            return jt.zeros((0, feat.shape[1], self.output_size, self.output_size), dtype=feat.dtype)
        pooled = []
        B, C, H, W = feat.shape
        for k in range(boxes.shape[0]):
            b, x1, y1, x2, y2 = boxes[k]
            b = int(b.item()) if hasattr(b, 'item') else int(b)
            # 转特征坐标并裁剪（确保切片范围非空）
            if W < 1 or H < 1:
                crop = feat[b:b+1, :, :, :]
                pooled.append(jt.nn.AdaptiveAvgPool2d((self.output_size, self.output_size))(crop))
                continue
            xs1 = max(int(math.floor(float(x1) / stride)), 0)
            ys1 = max(int(math.floor(float(y1) / stride)), 0)
            xs2i = int(math.ceil(float(x2) / stride))
            ys2i = int(math.ceil(float(y2) / stride))
            # 以右/下边界为开区间，允许等于尺寸
            xs2 = min(max(xs1 + 1, xs2i), W)
            ys2 = min(max(ys1 + 1, ys2i), H)
            # 若极端情况下依然无效，退化为整图
            if xs2 <= xs1 or ys2 <= ys1:
                crop = feat[b:b+1, :, :, :]
            else:
                crop = feat[b:b+1, :, ys1:ys2, xs1:xs2]
            # 若裁剪区域比输出更小，使用双线性插值上采样；否则使用自适应平均池化
            ch, cw = int(crop.shape[2]), int(crop.shape[3])
            if ch < self.output_size or cw < self.output_size:
                pooled.append(jt.nn.interpolate(crop, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False))
            else:
                pooled.append(jt.nn.AdaptiveAvgPool2d((self.output_size, self.output_size))(crop))
        return jt.concat(pooled, dim=0)

    def execute(self, feats, rois):
        # feats: list[Tensor] len=L, each [B, C, Hi, Wi]
        # rois: [N, 5] (b, x1, y1, x2, y2)
        assert isinstance(feats, (list, tuple))
        num_levels = len(feats)
        if rois.shape[0] == 0:
            return jt.zeros((0, feats[0].shape[1], self.output_size, self.output_size), dtype=feats[0].dtype)

        target_lvls = self._map_roi_levels(rois, num_levels)
        outputs = []
        for lvl in range(num_levels):
            nz = (target_lvls == lvl).nonzero()
            if nz.shape[0] == 0:
                continue
            idxs = nz.reshape(nz.shape[0])  # (M,)
            rois_lvl = rois[idxs, :]
            pooled = self._crop_and_resize(feats[lvl], rois_lvl, stride=self.featmap_strides[lvl])
            outputs.append((idxs, pooled))
        # 还原顺序
        if not outputs:
            return jt.zeros((0, feats[0].shape[1], self.output_size, self.output_size), dtype=feats[0].dtype)
        pooled_all = [None] * int(rois.shape[0])
        for idxs, pooled in outputs:
            for i_local in range(int(pooled.shape[0])):
                pooled_all[int(idxs[i_local].item())] = pooled[i_local:i_local+1]
        # 过滤 None（理论上不应存在）
        pooled_all = [p for p in pooled_all if p is not None]
        if len(pooled_all) == 0:
            return jt.zeros((0, feats[0].shape[1], self.output_size, self.output_size), dtype=feats[0].dtype)
        return jt.concat(pooled_all, dim=0)

