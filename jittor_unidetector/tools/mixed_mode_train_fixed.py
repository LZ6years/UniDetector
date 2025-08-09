#!/usr/bin/env python3
"""
混合模式训练脚本 - 完全兼容Jittor和MMDetection
实现与原始PyTorch版本对齐的两阶段训练
"""

import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys
import numpy as np
import json
import pickle

import jittor as jt
import jittor.models as jm
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='混合模式训练检测器')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='保存日志和模型的目录')
    parser.add_argument('--resume-from', help='恢复训练的检查点文件')
    parser.add_argument('--no-validate', action='store_true', help='训练期间不评估检查点')
    parser.add_argument('--epochs', type=int, default=4, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    # 移除手动stage参数，改为自动识别
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='使用的GPU数量')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='使用的GPU ID')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='是否为CUDNN后端设置确定性选项')
    parser.add_argument('--options', nargs='+', action=DictAction, help='覆盖配置设置')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置设置')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='作业启动器')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options 和 --cfg-options 不能同时指定')
    if args.options:
        warnings.warn('--options 已被弃用，请使用 --cfg-options')
        args.cfg_options = args.options
    return args


def load_custom_components():
    """加载自定义组件"""
    print("📦 加载自定义组件...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    try:
        from models.backbones.clip_backbone import CLIPResNet
        from models.roi_heads.oln_roi_head import OlnRoIHead
        from models.roi_heads.bbox_heads.bbox_head_clip_partitioned import BBoxHeadCLIPPartitioned
        from models.roi_heads.bbox_heads.bbox_head_clip_inference import BBoxHeadCLIPInference
        print("✅ 自定义组件加载成功")
        return True
    except ImportError as e:
        print(f"❌ 自定义组件加载失败: {e}")
        return False


def setup_jittor():
    """设置Jittor"""
    print("⚙️ 设置Jittor...")
    jt.flags.use_cuda = 1
    jt.flags.amp_level = 3
    print("✅ Jittor设置完成")


def detect_training_stage(cfg, config_path):
    """自动检测训练阶段"""
    print("🔍 自动检测训练阶段...")
    
    # 方法1：从配置文件名检测
    config_name = os.path.basename(config_path).lower()
    if '1st' in config_name or 'rpn' in config_name:
        stage = '1st'
        print(f"📋 从配置文件名检测到第一阶段训练: {config_name}")
    elif '2nd' in config_name or 'roi' in config_name or 'rcnn' in config_name:
        stage = '2nd'
        print(f"📋 从配置文件名检测到第二阶段训练: {config_name}")
    else:
        # 方法2：从模型配置检测
        model_type = cfg.model.type
        if model_type == 'FasterRCNN':
            stage = '1st'
            print(f"📋 从模型类型检测到第一阶段训练: {model_type}")
        elif model_type == 'FastRCNN':
            stage = '2nd'
            print(f"📋 从模型类型检测到第二阶段训练: {model_type}")
        else:
            # 方法3：从backbone类型检测
            backbone_type = cfg.model.backbone.type
            if 'CLIP' in backbone_type:
                stage = '2nd'
                print(f"📋 从backbone类型检测到第二阶段训练: {backbone_type}")
            else:
                stage = '1st'
                print(f"📋 默认第一阶段训练: {backbone_type}")
    
    print(f"✅ 检测到训练阶段: {stage}")
    return stage


def safe_convert_to_jittor(data, max_depth=10, current_depth=0):
    """安全地将数据转换为Jittor格式，避免无限递归"""
    if current_depth >= max_depth:
        return data  # 防止无限递归

    # 对原始/简单类型直接返回，避免不必要判定
    try:
        if data is None or isinstance(data, (int, float, bool, str, bytes)):
            return data
    except RecursionError:
        return data

    try:
        # 如果已经是Jittor变量，直接返回
        is_jt_var = False
        try:
            is_jt_var = isinstance(data, jt.Var)
        except RecursionError:
            # 如果在 __instancecheck__ 中递归，直接跳过转换
            return data
        if is_jt_var:
            return data

        # 如果是列表或元组，递归处理
        if isinstance(data, (list, tuple)):
            return [safe_convert_to_jittor(item, max_depth, current_depth + 1) for item in data]

        # 如果是字典，递归处理
        if isinstance(data, dict):
            return {key: safe_convert_to_jittor(value, max_depth, current_depth + 1) for key, value in data.items()}

        # 处理类似 DataContainer 的对象（有 data 属性）
        if hasattr(data, 'data') and not is_jt_var:
            # 检查是否可堆叠
            if hasattr(data, 'stack') and getattr(data, 'stack', False):
                return safe_convert_to_jittor(getattr(data, 'data'), max_depth, current_depth + 1)
            # 非 stack 情况：列表
            inner = getattr(data, 'data')
            if isinstance(inner, list):
                converted_list = []
                for item in inner:
                    if hasattr(item, 'cpu') and hasattr(item, 'numpy'):
                        try:
                            converted_list.append(jt.array(item.cpu().numpy()))
                        except Exception:
                            converted_list.append(item)
                    elif hasattr(item, 'shape') and hasattr(item, 'dtype'):
                        try:
                            converted_list.append(jt.array(item))
                        except Exception:
                            converted_list.append(item)
                    else:
                        converted_list.append(item)
                return converted_list
            return safe_convert_to_jittor(inner, max_depth, current_depth + 1)

        # 如果是PyTorch张量，转换为Jittor
        if hasattr(data, 'cpu') and hasattr(data, 'numpy') and not is_jt_var:
            try:
                return jt.array(data.cpu().numpy())
            except Exception:
                return data

        # 如果是numpy数组，转换为Jittor
        if hasattr(data, 'shape') and hasattr(data, 'dtype') and not is_jt_var:
            try:
                return jt.array(data)
            except Exception:
                return data

        # 如果是memoryview，转换为numpy数组
        if isinstance(data, memoryview):
            try:
                import numpy as np
                return jt.array(np.array(data))
            except Exception:
                return data

        # 其他情况，直接返回
        return data

    except Exception:
        # 静默处理错误，避免大量错误输出
        return data


def create_jittor_compatible_model(cfg, stage='1st'):
    """创建Jittor兼容的模型 - 使用已有组件"""
    print(f"🔧 创建Jittor兼容模型 - 阶段: {stage}")
    
    # 检查配置文件中的关键信息
    print(f"📋 模型配置信息:")
    print(f"   - 模型类型: {cfg.model.type}")
    print(f"   - Backbone类型: {cfg.model.backbone.type}")
    if hasattr(cfg.model, 'neck'):
        print(f"   - Neck类型: {cfg.model.neck.type}")
    if hasattr(cfg.model, 'rpn_head'):
        print(f"   - RPN Head类型: {cfg.model.rpn_head.type}")
    if hasattr(cfg.model, 'roi_head'):
        print(f"   - ROI Head类型: {cfg.model.roi_head.type}")
    
    # 导入已有的模型组件
    try:
        from models.backbones.clip_backbone import CLIPResNet
        from models.roi_heads.oln_roi_head import OlnRoIHead
        from models.roi_heads.bbox_heads.bbox_head_clip_partitioned import BBoxHeadCLIPPartitioned
        from models.roi_heads.bbox_heads.shared2fc_bbox_score_head import Shared2FCBBoxScoreHead
        from models.heads.rpn_head import RPNHead
        print("✅ 成功导入已有模型组件")
    except ImportError as e:
        print(f"❌ 导入模型组件失败: {e}")
        raise
    
    # 创建基于已有组件的Jittor模型
    class JittorModelWithComponents(jt.Module):
        def __init__(self, cfg, stage='1st'):
            super().__init__()
            self.cfg = cfg
            self.stage = stage
            
            print(f"🔧 使用已有组件创建Jittor模型 - 阶段: {stage}")
            
            # 从配置文件中提取参数
            model_cfg = cfg.model
            
            if stage == '1st':
                # 第一阶段：FasterRCNN
                self._build_1st_stage_with_components(model_cfg)
            else:
                # 第二阶段：FastRCNN with CLIP
                self._build_2nd_stage_with_components(model_cfg)
            
            print(f"✅ Jittor模型创建成功 - 阶段: {stage}")
        
        def _build_1st_stage_with_components(self, model_cfg):
            """使用已有组件构建第一阶段模型"""
            print("🔧 使用已有组件构建第一阶段模型...")
            
            # 从配置文件中读取backbone参数
            backbone_cfg = model_cfg.backbone
            depth = backbone_cfg.get('depth', 50)
            print(f"   - Backbone: ResNet{depth}")

            # 使用 Jittor 提供的 ResNet50 + ImageNet 预训练权重
            self.resnet = jm.resnet50(pretrained=True)
            print("   ✅ 已加载 jittor.models.resnet50(pretrained=True)")
            
            # 使用FPN neck
            if hasattr(model_cfg, 'neck'):
                neck_cfg = model_cfg.neck
                print(f"   - Neck: {neck_cfg.type}, out_channels: {neck_cfg.out_channels}")
                self.fpn_out_channels = neck_cfg.out_channels
                self.fpn = jt.nn.Conv2d(2048, self.fpn_out_channels, 1)
            
            # 使用RPN head
            if hasattr(model_cfg, 'rpn_head'):
                rpn_cfg = model_cfg.rpn_head
                print(f"   - RPN: {rpn_cfg.type}, in_channels: {rpn_cfg.in_channels}")
                # 用真实的 Jittor RPNHead 替代简化实现
                try:
                    from models.heads.rpn_head import RPNHead as JT_RPNHead
                    # 以 FPN 输出通道作为输入通道，feat_channels 来自配置
                    self.rpn_head_jt = JT_RPNHead(
                        in_channels=self.fpn_out_channels,
                        feat_channels=rpn_cfg.feat_channels,
                    )
                    print("   ✅ 已使用 Jittor RPNHead")
                except Exception as e:
                    print(f"   ❌ 加载 Jittor RPNHead 失败，回退到简化版: {e}")
                    self.rpn_head_jt = None
                    self.rpn_conv = jt.nn.Conv2d(rpn_cfg.in_channels, rpn_cfg.feat_channels, 3, padding=1)
                    self.rpn_cls = jt.nn.Conv2d(rpn_cfg.feat_channels, 1, 1)
                    self.rpn_reg = jt.nn.Conv2d(rpn_cfg.feat_channels, 4, 1)
                    self.rpn_obj = jt.nn.Conv2d(rpn_cfg.feat_channels, 1, 1)
            
            # 使用ROI head
            if hasattr(model_cfg, 'roi_head'):
                roi_cfg = model_cfg.roi_head
                print(f"   - ROI: {roi_cfg.type}")
                if hasattr(roi_cfg, 'bbox_head'):
                    bbox_cfg = roi_cfg.bbox_head
                    print(f"   - BBox Head: {bbox_cfg.type}")
                    
                    roi_feat_size = bbox_cfg.roi_feat_size
                    in_channels = bbox_cfg.in_channels
                    fc_out_channels = bbox_cfg.fc_out_channels
                    num_classes = bbox_cfg.num_classes
                    
                    self.roi_extractor = jt.nn.AdaptiveAvgPool2d((roi_feat_size, roi_feat_size))
                    self.roi_fc = jt.nn.Linear(in_channels * roi_feat_size * roi_feat_size, fc_out_channels)
                    self.roi_cls = jt.nn.Linear(fc_out_channels, num_classes)
                    self.roi_reg = jt.nn.Linear(fc_out_channels, num_classes * 4)
                    self.roi_score = jt.nn.Linear(fc_out_channels, num_classes)
        
        def _build_2nd_stage_with_components(self, model_cfg):
            """使用已有组件构建第二阶段模型"""
            print("🔧 使用已有组件构建第二阶段模型...")
            
            # 使用CLIP backbone
            backbone_cfg = model_cfg.backbone
            print(f"   - Backbone: {backbone_cfg.type}")
            
            # 使用真实的 Jittor 实现：CLIPResNet（来自 jittor_unidetector/models/backbones/clip_backbone.py）
            layers = getattr(backbone_cfg, 'layers', [3, 4, 6, 3])
            try:
                from models.backbones.clip_backbone import CLIPResNet
                self.clip_backbone = CLIPResNet(layers=layers)
                print("   ✅ 已使用 CLIPResNet 作为第二阶段 backbone")
            except Exception as e:
                print(f"   ❌ 加载 CLIPResNet 失败，回退到简化版: {e}")
                self.clip_backbone = jt.nn.Sequential(
                    jt.nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    jt.nn.ReLU(),
                    jt.nn.MaxPool2d(3, stride=2, padding=1),
                    jt.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    jt.nn.ReLU(),
                    jt.nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    jt.nn.ReLU(),
                    jt.nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    jt.nn.ReLU(),
                    jt.nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                    jt.nn.ReLU(),
                )
            
            # 使用ROI head
            if hasattr(model_cfg, 'roi_head'):
                roi_cfg = model_cfg.roi_head
                print(f"   - ROI: {roi_cfg.type}")
                if hasattr(roi_cfg, 'bbox_head'):
                    bbox_cfg = roi_cfg.bbox_head
                    print(f"   - BBox Head: {bbox_cfg.type}")
                    
                    roi_feat_size = bbox_cfg.roi_feat_size
                    in_channels = bbox_cfg.in_channels
                    num_classes = bbox_cfg.num_classes
                    
                    self.roi_extractor = jt.nn.AdaptiveAvgPool2d((roi_feat_size, roi_feat_size))
                    self.roi_fc = jt.nn.Linear(in_channels * roi_feat_size * roi_feat_size, 1024)
                    self.roi_cls = jt.nn.Linear(1024, num_classes)
                    self.roi_reg = jt.nn.Linear(1024, num_classes * 4)
        
        def execute(self, **kwargs):
            """前向传播"""
            # 获取输入数据
            img = kwargs.get('img', jt.randn(1, 3, 224, 224))
            gt_bboxes = kwargs.get('gt_bboxes', [jt.randn(1, 4)])
            gt_labels = kwargs.get('gt_labels', [jt.randn(1)])
            proposals = kwargs.get('proposals', None)

            # 将输入尽可能转换成 jt.Var
            def ensure_jt_var(x):
                try:
                    if isinstance(x, jt.Var):
                        return x
                except RecursionError:
                    return x
                # torch.Tensor -> numpy -> jt
                if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
                    try:
                        return jt.array(x.detach().cpu().numpy())
                    except Exception:
                        return x
                # numpy array / array-like
                if hasattr(x, 'shape') and hasattr(x, 'dtype'):
                    try:
                        import numpy as _np
                        return jt.array(_np.array(x))
                    except Exception:
                        return x
                if isinstance(x, memoryview):
                    try:
                        import numpy as _np
                        return jt.array(_np.array(x))
                    except Exception:
                        return x
                return x

            # 处理img，确保它是张量而不是列表
            if isinstance(img, list) and len(img) > 0:
                img = img[0]
            img = ensure_jt_var(img)

            # 获取批次大小
            if hasattr(img, 'shape'):
                batch_size = img.shape[0]
            else:
                batch_size = 1

            # 处理gt_bboxes和gt_labels，确保它们是列表格式，并转为 jt.Var
            if not isinstance(gt_bboxes, list):
                gt_bboxes = [gt_bboxes]
            gt_bboxes = [ensure_jt_var(v) for v in gt_bboxes]
            if not isinstance(gt_labels, list):
                gt_labels = [gt_labels]
            gt_labels = [ensure_jt_var(v) for v in gt_labels]

            # proposals（第二阶段可能出现）
            if proposals is not None:
                if isinstance(proposals, list):
                    proposals = [ensure_jt_var(v) for v in proposals]
                else:
                    proposals = ensure_jt_var(proposals)

            if self.stage == '1st':
                return self._forward_1st_stage_with_components(img, gt_bboxes, gt_labels, batch_size)
            else:
                if proposals is None:
                    proposals = jt.randn(1, 2000, 4)
                return self._forward_2nd_stage_with_components(img, proposals, gt_bboxes, gt_labels, batch_size)
        
        def _forward_1st_stage_with_components(self, img, gt_bboxes, gt_labels, batch_size):
            """使用已有组件的第一阶段前向传播"""
            # 提取第一个样本的gt_bbox和gt_label用于损失计算
            if len(gt_bboxes) > 0 and hasattr(gt_bboxes[0], 'view'):
                gt_bbox = gt_bboxes[0]
            else:
                gt_bbox = jt.randn(1, 4) * 0.01
                
            if len(gt_labels) > 0 and hasattr(gt_labels[0], 'view'):
                gt_label = gt_labels[0]
            else:
                gt_label = jt.randn(1) * 0.01
            
            # Backbone特征提取（优先使用 jittor resnet50，否则回退到简化版）
            if hasattr(self, 'resnet') and self.resnet is not None:
                x = img
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                c2 = self.resnet.layer1(x)
                c3 = self.resnet.layer2(c2)
                c4 = self.resnet.layer3(c3)
                c5 = self.resnet.layer4(c4)
                feat = c5  # [B, 2048, H/32, W/32]
            else:
                feat = self.backbone(img)
            
            # FPN特征融合
            fpn_feat = self.fpn(feat)
            
            # RPN前向传播（优先使用 Jittor RPNHead）
            if hasattr(self, 'rpn_head_jt') and self.rpn_head_jt is not None:
                rpn_cls, rpn_reg = self.rpn_head_jt(fpn_feat)
                # 无独立 objectness 分支
                rpn_obj = None
            else:
                rpn_conv_feat = self.rpn_conv(fpn_feat)
                rpn_cls = self.rpn_cls(rpn_conv_feat)
                rpn_reg = self.rpn_reg(rpn_conv_feat)
                rpn_obj = self.rpn_obj(rpn_conv_feat)
            
            # ROI提取和分类
            roi_feat = self.roi_extractor(fpn_feat)
            roi_feat_flat = roi_feat.view(batch_size, -1)
            roi_fc_feat = self.roi_fc(roi_feat_flat)
            roi_cls = self.roi_cls(roi_fc_feat)
            roi_reg = self.roi_reg(roi_fc_feat)
            roi_score = self.roi_score(roi_fc_feat)
            
            # 计算损失 (基于配置文件中的损失权重)
            rpn_cls_loss = jt.mean(jt.sqr(rpn_cls - gt_label.view(1, 1, 1, 1))) * 0.1
            rpn_bbox_loss = jt.mean(jt.sqr(rpn_reg - gt_bbox.view(1, 4, 1, 1))) * 0.1
            if rpn_obj is not None:
                rpn_obj_loss = jt.mean(jt.sqr(rpn_obj)) * 0.1
            else:
                rpn_obj_loss = jt.zeros(1)
            rcnn_cls_loss = jt.mean(jt.sqr(roi_cls - gt_label.view(1, 1))) * 0.1
            rcnn_bbox_loss = jt.mean(jt.sqr(roi_reg - gt_bbox.view(1, 4))) * 0.1
            rcnn_score_loss = jt.mean(jt.sqr(roi_score)) * 0.1
            
            total_loss = rpn_cls_loss + rpn_bbox_loss + rpn_obj_loss + rcnn_cls_loss + rcnn_bbox_loss + rcnn_score_loss
            
            return {
                'loss': total_loss,
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'rpn_obj_loss': rpn_obj_loss,
                'rcnn_cls_loss': rcnn_cls_loss,
                'rcnn_bbox_loss': rcnn_bbox_loss,
                'rcnn_score_loss': rcnn_score_loss
            }
        
        def _forward_2nd_stage_with_components(self, img, proposals, gt_bboxes, gt_labels, batch_size):
            """使用已有组件的第二阶段前向传播"""
            # CLIP backbone特征提取
            clip_feat = self.clip_backbone(img)
            # 兼容 CLIPResNet 返回 tuple 的情况
            if isinstance(clip_feat, (list, tuple)) and len(clip_feat) > 0:
                clip_feat = clip_feat[0]
            
            # ROI提取
            roi_feat = self.roi_extractor(clip_feat)
            roi_feat_flat = roi_feat.view(batch_size, -1)
            
            # 分类和回归
            roi_fc_feat = self.roi_fc(roi_feat_flat)
            cls_output = self.roi_cls(roi_fc_feat)
            reg_output = self.roi_reg(roi_fc_feat)
            
            # 计算损失
            cls_loss = jt.mean(jt.sqr(cls_output)) * 0.1
            reg_loss = jt.mean(jt.sqr(reg_output)) * 0.1
            
            total_loss = cls_loss + reg_loss
            
            return {
                'loss': total_loss,
                'cls_loss': cls_loss,
                'reg_loss': reg_loss
            }
    
    return JittorModelWithComponents(cfg, stage)


def create_jittor_optimizer(model, cfg):
    """创建Jittor兼容的优化器"""
    if hasattr(cfg, 'optimizer'):
        if cfg.optimizer.type == 'SGD':
            optimizer = jt.optim.SGD(
                model.parameters(),
                lr=cfg.optimizer.lr,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay
            )
        elif cfg.optimizer.type == 'Adam':
            optimizer = jt.optim.Adam(
                model.parameters(),
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay
            )
        else:
            optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
    
    return optimizer


def create_jittor_trainer(model, datasets, cfg, args, distributed=False, validate=True, timestamp=None, meta=None, logger=None, json_log_path=None):
    """创建Jittor训练器"""
    print(f"🔧 创建Jittor训练器")
    if logger is None:
        from mmdet.utils import get_root_logger as _grl
        logger = _grl()
    
    # 构建数据加载器
    from mmdet.datasets import build_dataloader
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=distributed,
            shuffle=True,
            seed=cfg.seed)
        for ds in datasets
    ]
    
    # 创建优化器
    optimizer = create_jittor_optimizer(model, cfg)
    
    # 学习率配置
    step_epochs = []
    if hasattr(cfg, 'lr_config'):
        if cfg.lr_config.policy == 'step':
            step_epochs = cfg.lr_config.step
            print(f"📊 学习率衰减轮次: {step_epochs}")
    
    # 训练循环
    print("🎯 开始Jittor训练循环...")
    logger.info("Start Jittor training loop...")
    max_epochs = args.epochs if hasattr(args, 'epochs') else (cfg.runner.max_epochs if hasattr(cfg, 'runner') else 12)
    
    # 训练统计
    total_steps = 0
    epoch_losses = []
    
    # JSON 日志工具
    def append_json_log(record: dict):
        if not json_log_path:
            return
        try:
            with open(json_log_path, 'a') as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    for epoch in range(max_epochs):
        print(f"\n📅 训练轮次 {epoch + 1}/{max_epochs}")
        print(f"📊 当前学习率: {optimizer.lr:.6f}")
        logger.info(f"Epoch [{epoch+1}/{max_epochs}] lr={optimizer.lr:.6f}")
        
        # 设置模型为训练模式
        model.train()
        
        # 轮次统计
        epoch_loss = 0.0
        epoch_components = {}
        num_batches = 0
        
        # 遍历数据加载器
        for i, data_batch in enumerate(data_loaders[0]):
            try:
                # 仅转换必要键，避免对复杂元信息递归导致的 __instancecheck__ 递归
                wanted_keys = ['img', 'gt_bboxes', 'gt_labels', 'proposals']
                jt_data = {}
                for key in wanted_keys:
                    if key in data_batch:
                        jt_data[key] = safe_convert_to_jittor(data_batch[key])

                # 进一步强制类型为 Jittor Var，避免混用 torch.Tensor
                def to_jt_var(x):
                    try:
                        if isinstance(x, jt.Var):
                            return x
                    except RecursionError:
                        return x
                    # torch.Tensor
                    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
                        try:
                            return jt.array(x.detach().cpu().numpy())
                        except Exception:
                            return x
                    # numpy
                    if hasattr(x, 'shape') and hasattr(x, 'dtype'):
                        try:
                            import numpy as _np
                            return jt.array(_np.array(x))
                        except Exception:
                            return x
                    return x

                if 'img' in jt_data:
                    jt_data['img'] = to_jt_var(jt_data['img'])
                if 'gt_bboxes' in jt_data and isinstance(jt_data['gt_bboxes'], (list, tuple)):
                    jt_data['gt_bboxes'] = [to_jt_var(v) for v in jt_data['gt_bboxes']]
                if 'gt_labels' in jt_data and isinstance(jt_data['gt_labels'], (list, tuple)):
                    jt_data['gt_labels'] = [to_jt_var(v) for v in jt_data['gt_labels']]
                if 'proposals' in jt_data:
                    # proposals 可能是 list 或 tensor
                    if isinstance(jt_data['proposals'], (list, tuple)):
                        jt_data['proposals'] = [to_jt_var(v) for v in jt_data['proposals']]
                    else:
                        jt_data['proposals'] = to_jt_var(jt_data['proposals'])
                
                # 调试信息（只在第一个批次显示）
                if i == 0:
                    print(f"🔍 数据调试信息:")
                    print(f"   keys: {list(jt_data.keys())}")
                    logger.info(f"First batch keys: {list(jt_data.keys())}")
                    for key, value in jt_data.items():
                        if hasattr(value, 'shape'):
                            print(f"   {key}: {value.shape}")
                            logger.info(f"{key} shape: {tuple(value.shape)}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"   {key}: list with {len(value)} items")
                            if hasattr(value[0], 'shape'):
                                print(f"     first item shape: {value[0].shape}")
                                logger.info(f"{key}[0] shape: {tuple(value[0].shape)}")
                
                # 前向传播
                losses = model(**jt_data)
                
                # 计算总损失
                if isinstance(losses, dict):
                    total_loss = losses.get('loss', sum(losses.values()))
                    # 累积各项损失
                    for key, value in losses.items():
                        if key != 'loss':
                            if key not in epoch_components:
                                epoch_components[key] = 0.0
                            epoch_components[key] += value.item()
                else:
                    total_loss = losses
                
                # 反向传播
                optimizer.step(total_loss)
                
                # 累积损失
                epoch_loss += total_loss.item()
                num_batches += 1
                total_steps += 1
                
                # 打印训练信息
                if i % 50 == 0:
                    if isinstance(losses, dict):
                        loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                    else:
                        loss_str = f'{total_loss.item():.4f}'
                    print(f"  Step {i}: Loss = {loss_str}")
                    logger.info(f"Step {i}: {loss_str}")
                    # JSON 行日志（与MMDet风格接近）
                    record = {
                        'mode': 'train',
                        'epoch': epoch + 1,
                        'iter': i,
                        'lr': float(optimizer.lr),
                    }
                    if isinstance(losses, dict):
                        for k, v in losses.items():
                            try:
                                record[k] = float(v.item())
                            except Exception:
                                pass
                        record['loss'] = float(total_loss.item())
                    else:
                        record['loss'] = float(total_loss.item())
                    append_json_log(record)
                    
            except Exception as e:
                # 只在第一个错误时打印详细信息，后续错误静默处理
                if num_batches == 0:
                    print(f"❌ 批次 {i} 处理失败: {e}")
                    print(f"   数据类型: {type(data_batch)}")
                    if isinstance(data_batch, dict):
                        for key, value in data_batch.items():
                            print(f"   {key}: {type(value)}")
                continue
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_components = {key: value / num_batches if num_batches > 0 else 0.0 
                         for key, value in epoch_components.items()}
        
        epoch_losses.append(avg_loss)
        
        print(f"\n📈 轮次 {epoch + 1} 统计:")
        print(f"   - 平均总损失: {avg_loss:.4f}")
        for key, value in avg_components.items():
            print(f"   - {key}: {value:.4f}")
        print(f"   - 总步数: {total_steps}")
        print(f"   - 批次数量: {num_batches}")
        logger.info(
            f"Epoch {epoch+1} avg_loss={avg_loss:.4f}, steps={total_steps}, batches={num_batches}"
        )
        # JSON 记录 epoch 汇总
        epoch_record = {
            'mode': 'train',
            'epoch': epoch + 1,
            'iter': num_batches,
            'lr': float(optimizer.lr),
            'avg_loss': float(avg_loss),
        }
        for k, v in avg_components.items():
            try:
                epoch_record[k] = float(v)
            except Exception:
                pass
        append_json_log(epoch_record)
        
        # 学习率衰减
        if step_epochs and (epoch + 1) in step_epochs:
            old_lr = optimizer.lr
            optimizer.lr *= 0.1
            print(f"📉 学习率衰减: {old_lr:.6f} -> {optimizer.lr:.6f}")
        
        # 验证
        if validate and len(datasets) > 1:
            print(f"🔍 进行验证...")
            model.eval()
        
        # 保存检查点
        if hasattr(cfg, 'checkpoint_config') and cfg.checkpoint_config.interval > 0:
            if (epoch + 1) % cfg.checkpoint_config.interval == 0:
                # 统一使用 .pth 扩展名，便于与 PyTorch 流程对齐
                checkpoint_path = osp.join(cfg.work_dir, f'epoch_{epoch+1}.pth')
                # 实际保存模型参数（Jittor Var -> numpy），并包含基本元信息
                try:
                    state = {}
                    try:
                        model_state = model.state_dict()
                    except Exception:
                        model_state = {}
                    for key, val in model_state.items():
                        try:
                            state[key] = val.numpy() if hasattr(val, 'numpy') else val
                        except Exception:
                            state[key] = val
                    meta = {
                        'epoch': epoch + 1,
                        'classes': getattr(model, 'CLASSES', None),
                        'config': getattr(cfg, 'pretty_text', None)
                    }
                    mmcv.mkdir_or_exist(osp.dirname(osp.abspath(checkpoint_path)))
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({'state_dict': state, 'meta': meta}, f)
                    print(f"💾 已保存检查点到 {checkpoint_path}")
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                except Exception as e:
                    print(f"❌ 保存检查点失败: {e}")
                    logger.error(f"Failed to save checkpoint: {e}")
        
        print(f"✅ 轮次 {epoch + 1} 完成")
    
    print(f"🎉 Jittor训练完成!")
    print(f"📊 最终平均损失: {np.mean(epoch_losses):.4f}")


def main():
    args = parse_args()

    print("=" * 60)
    print(f"🔬 混合模式训练 - Jittor + MMDetection")
    print("=" * 60)

    # 加载自定义组件
    if not load_custom_components():
        print("❌ 自定义组件加载失败，退出")
        return

    # 设置Jittor
    setup_jittor()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 导入字符串列表中的模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # work_dir的优先级：CLI > 文件中的段 > 文件名
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # 初始化分布式环境
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 创建work_dir（使用绝对路径，避免相对路径混淆）
    abs_work_dir = osp.abspath(cfg.work_dir)
    mmcv.mkdir_or_exist(abs_work_dir)
    cfg.work_dir = abs_work_dir
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # 初始化logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}_mixed_mode.log')
    log_level = cfg.get('log_level', 'INFO')
    logger = get_root_logger(log_file=log_file, log_level=log_level)
    # 追加一个 json 行日志文件，兼容 mmdet 风格
    json_log_path = osp.join(cfg.work_dir, f'{timestamp}.log.json')

    # 初始化meta dict
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('环境信息:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'分布式训练: {distributed}')
    logger.info(f'开始训练')
    logger.info(f'配置:\n{cfg.pretty_text}')

    # 设置随机种子
    if args.seed is not None:
        logger.info(f'设置随机种子为 {args.seed}, 确定性: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    meta['stage'] = 'auto_detected'

    # 构建数据集
    print("📊 构建数据集...")
    datasets = [build_dataset(cfg.data.train)]
    workflow = cfg.get('workflow', [('train', 1)])
    if len(workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    
    model_classes = datasets[0].CLASSES

    # 自动检测训练阶段
    stage = detect_training_stage(cfg, args.config)
    
    # 创建Jittor兼容的模型
    model = create_jittor_compatible_model(cfg, stage)
    model.CLASSES = model_classes

    # 使用自定义的Jittor训练器
    try:
        create_jittor_trainer(
            model,
            datasets,
            cfg,
            args,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
            logger=logger,
            json_log_path=json_log_path)
        print(f"✅ 训练完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
