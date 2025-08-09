import json
import numpy as np
from .base_dataset import BaseDataset, Compose
import os
import cv2

class COCODataset(BaseDataset):
    """
    COCO数据集类
    """
    
    def __init__(self, ann_file, img_prefix, pipeline=None, test_mode=False, 
                 filter_empty_gt=True, min_size=32):
        super(COCODataset, self).__init__(ann_file, img_prefix, pipeline, test_mode)
        
        self.filter_empty_gt = filter_empty_gt
        self.min_size = min_size
        
        # 构建数据增强管道
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        
        # 过滤空标注
        if self.filter_empty_gt:
            self.data_infos = self._filter_empty_gt()
    
    def load_annotations(self, ann_file):
        """加载COCO标注文件"""
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        # 获取图像信息
        self.ids = list(sorted(self.coco['images']))
        
        # 获取类别信息
        self.cat_ids = self.coco.get('categories', [])
        self.cat2label = {cat['id']: i for i, cat in enumerate(self.cat_ids)}
        
        # 构建图像到标注的映射
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        return self.ids
    
    def get_ann_info(self, idx):
        """获取标注信息"""
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        
        return self._parse_ann_info(ann_info)
    
    def _parse_ann_info(self, ann_info):
        """解析标注信息"""
        gt_bboxes = []
        gt_labels = []
        
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            
            x1, y1, w, h = ann['bbox']
            if w < self.min_size or h < self.min_size:
                continue
            
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
        
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        
        ann = {
            'bboxes': gt_bboxes,
            'labels': gt_labels,
        }
        
        return ann
    
    def _filter_empty_gt(self):
        """过滤空标注的图像"""
        valid_inds = []
        for i, img_id in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            ann_info = self.coco.loadAnns(ann_ids)
            
            if len(ann_info) > 0:
                valid_inds.append(i)
        
        return [self.ids[i] for i in valid_inds]
    
    def __getitem__(self, idx):
        """获取单个样本"""
        data_info = self.data_infos[idx]
        
        # 加载图像
        img = self.load_image(data_info)
        
        # 获取标注信息
        ann_info = self.get_ann_info(idx)
        
        # 构建数据字典
        data = {
            'img': img,
            'img_info': data_info,
            'gt_bboxes': ann_info['bboxes'],
            'gt_labels': ann_info['labels'],
            'img_metas': {
                'filename': data_info['file_name'],
                'ori_shape': img.shape,
                'img_shape': img.shape,
                'pad_shape': img.shape,
                'scale_factor': 1.0,
                'flip': False,
                'flip_direction': None
            }
        }
        
        # 应用数据增强管道
        if self.pipeline is not None:
            data = self.pipeline(data)
        
        return data
    
    def load_image(self, data_info):
        """加载图像"""
        img_path = os.path.join(self.img_prefix, data_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_cat_ids(self, idx):
        """获取类别ID"""
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]
    
    def evaluate(self, results, metric='bbox', logger=None):
        """评估结果"""
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        
        if len(results) != len(self):
            raise ValueError(f'The length of results is not equal to the dataset len: '
                           f'{len(results)} != {len(self)}')
        
        # 这里可以实现COCO评估逻辑
        # 为了简化，我们返回一个基本的评估结果
        eval_results = {
            'bbox_mAP': 0.0,
            'bbox_mAP_50': 0.0,
            'bbox_mAP_75': 0.0,
        }
        
        return eval_results 