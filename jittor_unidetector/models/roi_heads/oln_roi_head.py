import jittor as jt
import jittor.nn as nn
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.builder import HEADS

@HEADS.register_module(force=True)
class OlnRoIHead(StandardRoIHead):
    """
    OLN (Object Localization Network) RoI头
    用于第一阶段的类无关区域提议
    参考UniDetector的OlnRoIHead实现
    """
    
    def __init__(self, **kwargs):
        super(OlnRoIHead, self).__init__(**kwargs)
    
    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, **kwargs):
        """
        训练时前向传播
        Args:
            x: 特征
            img_metas: 图像元信息
            proposal_list: 提议列表
            gt_bboxes: 真实边界框
            gt_labels: 真实标签
        Returns:
            losses: 损失字典
        """
        # 初始化损失字典
        losses = {}
        
        # 分配目标
        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        
        # 提取RoI特征
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])
        
        return losses
    
    def _bbox_forward_train(self, x, sampling_results):
        """边界框前向训练"""
        rois = jt.concat([res.bboxes for res in sampling_results])
        bbox_results = self.bbox_head(x, rois)
        
        bbox_targets = self.bbox_head.get_target(sampling_results, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """
        简单测试
        Args:
            x: 特征
            proposal_list: 提议列表
            img_metas: 图像元信息
            rescale: 是否重新缩放
        Returns:
            bbox_results: 边界框结果
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        
        return bbox_results
    
    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """简单测试边界框"""
        rois = jt.concat(proposals)
        bbox_results = self.bbox_head(x, rois)
        
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois, bbox_results['cls_score'], bbox_results['bbox_pred'],
            img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
        
        return det_bboxes, det_labels


def bbox2result(bboxes, labels, num_classes):
    """将边界框转换为结果格式"""
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)] 