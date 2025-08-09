import jittor as jt
import numpy as np
import cv2
import os
from abc import ABC, abstractmethod

class BaseDataset(jt.dataset.Dataset, ABC):
    """
    基础数据集类
    """
    
    def __init__(self, ann_file, img_prefix, pipeline=None, test_mode=False):
        super(BaseDataset, self).__init__()
        
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline or []
        self.test_mode = test_mode
        
        # 加载数据
        self.data_infos = self.load_annotations(ann_file)
        
    @abstractmethod
    def load_annotations(self, ann_file):
        """加载标注文件"""
        pass
    
    @abstractmethod
    def get_ann_info(self, idx):
        """获取标注信息"""
        pass
    
    def __len__(self):
        return len(self.data_infos)
    
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
                'filename': data_info['filename'],
                'ori_shape': img.shape,
                'img_shape': img.shape,
                'pad_shape': img.shape,
                'scale_factor': 1.0,
                'flip': False,
                'flip_direction': None
            }
        }
        
        # 应用数据增强管道
        data = self.pipeline(data)
        
        return data
    
    def load_image(self, data_info):
        """加载图像"""
        img_path = os.path.join(self.img_prefix, data_info['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

class Compose:
    """数据增强管道"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class LoadImageFromFile:
    """从文件加载图像"""
    
    def __call__(self, data):
        img = data['img']
        data['img'] = img
        return data

class LoadAnnotations:
    """加载标注"""
    
    def __init__(self, with_bbox=True, with_label=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
    
    def __call__(self, data):
        if self.with_bbox:
            data['gt_bboxes'] = data['gt_bboxes']
        if self.with_label:
            data['gt_labels'] = data['gt_labels']
        return data

class Resize:
    """调整图像大小"""
    
    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio
    
    def __call__(self, data):
        img = data['img']
        h, w = img.shape[:2]
        
        if isinstance(self.img_scale, tuple):
            target_h, target_w = self.img_scale
        else:
            target_h, target_w = self.img_scale
        
        if self.keep_ratio:
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = target_h, target_w
        
        img = cv2.resize(img, (new_w, new_h))
        data['img'] = img
        data['img_metas']['img_shape'] = img.shape
        data['img_metas']['scale_factor'] = scale if self.keep_ratio else 1.0
        
        return data

class RandomFlip:
    """随机翻转"""
    
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio
    
    def __call__(self, data):
        if np.random.random() < self.flip_ratio:
            img = data['img']
            img = cv2.flip(img, 1)  # 水平翻转
            data['img'] = img
            data['img_metas']['flip'] = True
            data['img_metas']['flip_direction'] = 'horizontal'
        return data

class Normalize:
    """归一化"""
    
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    
    def __call__(self, data):
        img = data['img'].astype(np.float32)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        img = (img - self.mean) / self.std
        data['img'] = img
        return data

class Pad:
    """填充"""
    
    def __init__(self, size_divisor=32):
        self.size_divisor = size_divisor
    
    def __call__(self, data):
        img = data['img']
        h, w = img.shape[:2]
        
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        data['img'] = img
        data['img_metas']['pad_shape'] = img.shape
        return data

class DefaultFormatBundle:
    """默认格式打包"""
    
    def __call__(self, data):
        # 转换为Jittor张量
        data['img'] = jt.array(data['img']).transpose(2, 0, 1)  # HWC -> CHW
        data['gt_bboxes'] = jt.array(data['gt_bboxes'])
        data['gt_labels'] = jt.array(data['gt_labels'])
        return data

class Collect:
    """收集指定键"""
    
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        return {key: data[key] for key in self.keys} 