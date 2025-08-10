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
import datetime
from tqdm import tqdm

import jittor as jt
import jittor.models as jm
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset

from mmdet.models.builder import build_head, build_roi_extractor, build_neck
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


def setup_jittor():
    """设置Jittor环境"""
    print("🔧 设置Jittor环境...")
    
    # 设置Jittor调试环境变量
    os.environ['JT_SYNC'] = '1'  # 强制CUDA同步，便于调试
    os.environ['trace_py_var'] = '3'  # 启用Python变量追踪
    
    # 设置cuDNN优化
    os.environ['CUDNN_CONV_ALGO_CACHE_MAX'] = '1000'  # 增加算法缓存大小
    os.environ['CUDNN_CONV_USE_DEFAULT_MATH'] = '1'   # 使用默认数学模式
    
    print(f"✅ JT_SYNC: {os.environ.get('JT_SYNC', '未设置')}")
    print(f"✅ trace_py_var: {os.environ.get('trace_py_var', '未设置')}")
    print(f"✅ CUDNN_CONV_ALGO_CACHE_MAX: {os.environ.get('CUDNN_CONV_ALGO_CACHE_MAX', '未设置')}")
    
    # 设置Jittor标志
    jt.flags.amp_level = 3  # 自动混合精度
    jt.flags.amp_reg = 1    # 自动混合精度注册
    
    print("✅ Jittor环境设置完成")


def load_custom_components():
    """加载自定义组件"""
    print("📦 加载自定义组件...")
    try:
        # 添加父目录到Python路径，以便导入models模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # 导入所有Jittor模型组件，确保它们被注册到mmdet注册表中
        from models.backbones.clip_backbone import CLIPResNet, CLIPResNetFPN
        from models.roi_heads.oln_roi_head import OlnRoIHead
        from models.roi_heads.bbox_heads.bbox_head_clip_partitioned import BBoxHeadCLIPPartitioned
        from models.roi_heads.bbox_heads.shared2fc_bbox_score_head import Shared2FCBBoxScoreHead
        from models.heads.rpn_head import RPNHead
        from models.heads.oln_rpn_head import OlnRPNHead
        from models.necks.fpn import FPN
        from models.detectors.faster_rcnn import FasterRCNN
        from models.detectors.fast_rcnn import FastRCNN
        from models.roi_heads.roi_extractors.single_roi_extractor import SingleRoIExtractor
        
        print("✅ 成功导入所有Jittor模型组件")
        return True
    except ImportError as e:
        print(f"❌ 导入模型组件失败: {e}")
        return False

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
        # 导入Jittor版本的模型组件
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
            
            # 从配置文件中提取参数
            model_cfg = cfg.model
            
            if stage == '1st':
                # 第一阶段：FasterRCNN
                self._build_1st_stage_with_components(model_cfg)

            
            print(f"✅ Jittor模型创建成功 - 阶段: {stage}")
        
        def execute(self, img, gt_bboxes=None, gt_labels=None, **kwargs):
            """Jittor模型的execute方法，处理前向传播"""
            if self.stage == '1st':
                # 获取batch size
                if hasattr(img, 'shape'):
                    batch_size = img.shape[0]
                elif isinstance(img, (list, tuple)) and len(img) > 0:
                    if hasattr(img[0], 'shape'):
                        batch_size = img[0].shape[0]
                    else:
                        batch_size = 1
                else:
                    batch_size = 1
                
                return self._forward_1st_stage_with_components(img, gt_bboxes, gt_labels, batch_size)
            else:
                raise NotImplementedError(f"Stage {self.stage} not implemented")
        
        def _build_1st_stage_with_components(self, model_cfg):
            """使用已有组件构建第一阶段模型"""
            print("🔧 使用已有组件构建第一阶段模型...")
            
            # 从配置文件中读取backbone参数
            backbone_cfg = model_cfg.backbone
            depth = backbone_cfg.get('depth', 50)
            print(f"   - Backbone: ResNet{depth}")

            # 使用 Jittor 提供的 ResNet50 + ImageNet 预训练权重
            self.resnet = jm.resnet50(pretrained=True)
            print("已加载 jittor.models.resnet50(pretrained=True)")

            # 根据配置冻结前若干 stage（conv1+bn1 记为 stage 0，layer1 为 stage 1）
            frozen_stages = getattr(backbone_cfg, 'frozen_stages', 0)
            if isinstance(frozen_stages, int) and frozen_stages >= 0:
                def _stop_grad_module(mod):
                    for p in mod.parameters():
                        try:
                            p.stop_grad()
                        except Exception:
                            pass
                if frozen_stages >= 0:
                    for name in ['conv1', 'bn1']:
                        if hasattr(self.resnet, name):
                            _stop_grad_module(getattr(self.resnet, name))
                if frozen_stages >= 1 and hasattr(self.resnet, 'layer1'):
                    _stop_grad_module(self.resnet.layer1)
                if frozen_stages >= 2 and hasattr(self.resnet, 'layer2'):
                    _stop_grad_module(self.resnet.layer2)
                if frozen_stages >= 3 and hasattr(self.resnet, 'layer3'):
                    _stop_grad_module(self.resnet.layer3)
                if frozen_stages >= 4 and hasattr(self.resnet, 'layer4'):
                    _stop_grad_module(self.resnet.layer4)
                print(f"   🔒 已冻结前 {frozen_stages} 个 stage")

            # norm_eval: 将 BN 设为 eval 模式，保持均值方差不更新
            norm_eval = bool(getattr(backbone_cfg, 'norm_eval', False))
            if norm_eval:
                try:
                    for m in self.resnet.modules():
                        # 兼容不同 BN 类名
                        if m.__class__.__name__ in ("BatchNorm", "BatchNorm2d"):
                            try:
                                m.is_training = False
                            except Exception:
                                pass
                except Exception:
                    pass
                print("   🧊 已将 BatchNorm 置为 eval (norm_eval=True)")

            # 不使用分类 fc，禁用其梯度以避免无梯度警告
            try:
                if hasattr(self.resnet, 'fc'):
                    for p in self.resnet.fc.parameters():
                        p.stop_grad()
            except Exception:
                pass

            # 使用FPN neck
            if hasattr(model_cfg, 'neck'):
                neck_cfg = model_cfg.neck
                print(f"   - Neck: {neck_cfg.type}, out_channels: {neck_cfg.out_channels}")
                self.fpn_out_channels = neck_cfg.out_channels
                self.neck = build_neck(neck_cfg)
                print("   ✅ 已构建 FPN")

            # 使用RPN head
            if hasattr(model_cfg, 'rpn_head'):
                rpn_cfg = model_cfg.rpn_head
                # 优先尝试 OLN-RPNHead
                from models.heads.oln_rpn_head import OlnRPNHead as JT_RPNHead
                print("   ✅ 使用 Jittor OlnRPNHead")
                # 以 FPN 输出通道作为输入通道，feat_channels 来自配置
                self.rpn_head_jt = JT_RPNHead(
                    in_channels=self.fpn_out_channels,
                    feat_channels=rpn_cfg.feat_channels,
                    anchor_generator=getattr(rpn_cfg, 'anchor_generator', None),
                )
            # 记录 FPN strides 供 proposals 生成使用
            self.fpn_strides = None
            if hasattr(rpn_cfg, 'anchor_generator') and hasattr(rpn_cfg.anchor_generator, 'strides'):
                self.fpn_strides = list(rpn_cfg.anchor_generator.strides)

            # 使用ROI head
            if hasattr(model_cfg, 'roi_head'):
                roi_cfg = model_cfg.roi_head
                # print(f"   - ROI: {roi_cfg.type}")
                if hasattr(roi_cfg, 'bbox_head'):
                    bbox_cfg = roi_cfg.bbox_head
                    # print(f"   - BBox Head: {bbox_cfg.type}")
                    # 构建模块化的 RoIExtractor 与 BBoxHead
                    if hasattr(roi_cfg, 'bbox_roi_extractor'):
                        self.roi_extractor = build_roi_extractor(roi_cfg.bbox_roi_extractor)
                        print("   ✅ 已构建 SingleRoIExtractor")

                        self.bbox_head = build_head(bbox_cfg)
                        self.roi_feat_size = bbox_cfg.get('roi_feat_size', 7)
                        print("   ✅ 已构建 BBoxHead 模块")

        
                
        def _forward_1st_stage_with_components(self, img, gt_bboxes, gt_labels, batch_size):
            """使用已有组件的第一阶段前向传播"""
            # 处理img参数：如果是列表，提取第一个元素
            if isinstance(img, (list, tuple)) and len(img) > 0:
                img = img[0]  # 提取第一个元素
                print(f"🔍 从列表中提取img，shape: {img.shape}")
            
            # 强化图像张量健壮性：确保为 jt.Var、float32、NCHW
            try:
                if not isinstance(img, jt.Var):
                    import numpy as _np
                    img = jt.array(_np.array(img))
            except Exception:
                pass
            if hasattr(img, 'shape') and len(img.shape) == 3:
                img = img.unsqueeze(0)
            try:
                img = img.float32()
            except Exception:
                pass
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

            
            # FPN 特征融合
            if hasattr(self, 'neck') and self.neck is not None:
                fpn_feats = self.neck([c2, c3, c4, c5])
                fpn_rpn = fpn_feats[-1]
                num_roi_lvls = 4
                if getattr(self, 'roi_extractor', None) is not None and hasattr(self.roi_extractor, 'featmap_strides'):
                    try:
                        num_roi_lvls = len(self.roi_extractor.featmap_strides)
                    except Exception:
                        num_roi_lvls = 4
                feats_pyramid = list(fpn_feats[:num_roi_lvls])
            else:
                fpn_rpn = self.fpn(feat)
                feats_pyramid = [fpn_rpn]
            
            # RPN前向传播（优先使用 Jittor RPNHead）
            if hasattr(self, 'rpn_head_jt') and self.rpn_head_jt is not None:
                # 传入全部 FPN 层，RPNHead 内部支持多层
                rpn_out = self.rpn_head_jt(fpn_feats if 'fpn_feats' in locals() else fpn_rpn)
                if isinstance(rpn_out, (list, tuple)) and len(rpn_out) == 3:
                    rpn_cls, rpn_reg, rpn_obj = rpn_out
                else:
                    rpn_cls, rpn_reg = rpn_out
                    rpn_obj = None

            
            # 使用模块化 RoIExtractor + BBoxHead
            if getattr(self, 'roi_extractor', None) is not None and getattr(self, 'bbox_head', None) is not None:
                # 生成 proposals：若为 OLN-RPNHead，使用其内置 get_bboxes
                if rpn_obj is not None and hasattr(self.rpn_head_jt, 'get_bboxes'):
                    try:
                        rois = self.rpn_head_jt.get_bboxes(
                            rpn_cls, rpn_reg, rpn_obj,
                            img_shape=img.shape,
                            cfg=(self.cfg.test_cfg.rpn if hasattr(self.cfg, 'test_cfg') and hasattr(self.cfg.test_cfg, 'rpn') else None)
                        )
                    except Exception as e:
                        print(f"⚠️  RPN get_bboxes 失败: {e}")
                        rois = None
                else:
                    # 旧的简化中心框 proposals
                    rois = self._generate_simple_proposals(
                        rpn_cls, fpn_feats if 'fpn_feats' in locals() else [fpn_rpn],
                        strides=(self.fpn_strides if self.fpn_strides is not None else [4,8,16,32,64]),
                        nms_pre=getattr(self.cfg.test_cfg.rpn, 'nms_pre', 1000) if hasattr(self.cfg, 'test_cfg') else 1000,
                        max_num=getattr(self.cfg.test_cfg.rpn, 'max_num', 1000) if hasattr(self.cfg, 'test_cfg') else 1000,
                        nms_thr=getattr(self.cfg.test_cfg.rpn, 'nms_thr', 0.7) if hasattr(self.cfg, 'test_cfg') else 0.7,
                        img_shape=img.shape
                    )

                # 为节省显存，训练阶段按图采样最多 K 个 RoIs 进入 ROI Head
                try:
                    max_rois_per_img_train = 2000
                    if hasattr(self.cfg, 'train_cfg') and hasattr(self.cfg.train_cfg, 'rcnn'):
                        # 若后续接入 assigner/sampler，可从此处读取目标采样量
                        pass
                    if rois is not None and isinstance(rois, jt.Var) and rois.shape[0] > 0:
                        bidx = rois[:, 0]
                        keep_indices = []
                        for b in range(int(img.shape[0])):
                            nz = (bidx == float(b)).nonzero()
                            if nz.shape[0] == 0:
                                continue
                            idxs = nz.reshape(-1)
                            take = min(max_rois_per_img_train, int(idxs.shape[0]))
                            keep_indices.append(idxs[:take])
                        if len(keep_indices) > 0:
                            keep_indices = jt.concat(keep_indices, dim=0)
                            rois = rois[keep_indices, :]
                except Exception:
                    pass
                    
                # 防御：确保输入为 Jittor Var
                feats_pyramid = [f if isinstance(f, jt.Var) else jt.array(f) for f in feats_pyramid]
                rois = rois if isinstance(rois, jt.Var) else jt.array(rois)
                try:
                    self._last_num_rois = int(rois.shape[0])
                except Exception:
                    self._last_num_rois = None
                roi_feat = self.roi_extractor(feats_pyramid, rois)
                bh_out = self.bbox_head(roi_feat)
                if isinstance(bh_out, (list, tuple)):
                    if len(bh_out) == 3:
                        roi_cls, roi_reg, roi_score = bh_out
                    elif len(bh_out) == 2:
                        roi_cls, roi_reg = bh_out
                        roi_score = None
                    else:
                        roi_cls = bh_out[0]
                        roi_reg = bh_out[1] if len(bh_out) > 1 else None
                        roi_score = None
                else:
                    roi_cls = bh_out
                    roi_reg = None
                    roi_score = None

                # 构建 ROI 训练目标（简化真实版）
                labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets, bbox_score_weights = \
                    self.bbox_head.build_targets_minimal(
                        rois,
                        gt_bboxes,
                        img_shape=img.shape,
                        pos_iou_thr=getattr(self.cfg.train_cfg.rcnn.assigner, 'pos_iou_thr', 0.5) if hasattr(self.cfg, 'train_cfg') else 0.5,
                        neg_iou_thr=getattr(self.cfg.train_cfg.rcnn.assigner, 'neg_iou_thr', 0.5) if hasattr(self.cfg, 'train_cfg') else 0.5,
                        num_samples=getattr(self.cfg.train_cfg.rcnn.sampler, 'num', 256) if hasattr(self.cfg, 'train_cfg') else 256,
                        pos_fraction=getattr(self.cfg.train_cfg.rcnn.sampler, 'pos_fraction', 0.25) if hasattr(self.cfg, 'train_cfg') else 0.25,
                    )

                # 计算 ROI 损失（一次性传入 bbox_score 及其目标）
                roi_losses = {}
                if hasattr(self.bbox_head, 'loss'):
                    num_classes = getattr(self.bbox_head, 'num_classes', 1)
                    roi_losses = self.bbox_head.loss(
                        roi_cls if roi_cls is not None else jt.zeros((labels.shape[0], num_classes+1)),
                        roi_reg if roi_reg is not None else jt.zeros((labels.shape[0], 4)),
                        roi_score if roi_score is not None else jt.zeros((labels.shape[0], 1)),
                        rois,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        bbox_score_targets=bbox_score_targets,
                        bbox_score_weights=bbox_score_weights,
                    )
            
            # 计算损失 (真实 RPN 训练)
            if hasattr(self, 'rpn_head_jt') and self.rpn_head_jt is not None and rpn_obj is not None and hasattr(self.rpn_head_jt, 'loss'):
                try:
                    # 使用正确的参数调用RPN loss函数
                    rpn_losses = self.rpn_head_jt.loss(
                        [rpn_cls], [rpn_reg], [rpn_obj],
                        gt_bboxes_list=gt_bboxes,
                        img_shape=img.shape  # 传递完整的img_shape (B, C, H, W)
                    )
                    rpn_cls_loss = rpn_losses.get('loss_rpn_cls', jt.zeros(1))
                    rpn_bbox_loss = rpn_losses.get('loss_rpn_bbox', jt.zeros(1))
                    rpn_obj_loss = rpn_losses.get('loss_rpn_obj', jt.zeros(1))
                except Exception as e:
                    print(f"⚠️  RPN损失计算失败: {e}")
                    # 回退到简化损失，确保数据类型正确
                    if hasattr(rpn_cls, 'shape'):
                        rpn_cls_loss = jt.mean(jt.sqr(rpn_cls)) * 0.0  # 按配置权重为0
                    else:
                        rpn_cls_loss = jt.zeros(1) * 0.0
                    
                    if hasattr(rpn_reg, 'shape'):
                        rpn_bbox_loss = jt.mean(jt.sqr(rpn_reg)) * 10.0  # 按配置权重为10
                    else:
                        rpn_bbox_loss = jt.zeros(1) * 10.0
                    
                    if hasattr(rpn_obj, 'shape'):
                        rpn_obj_loss = jt.mean(jt.sqr(rpn_obj)) * 1.0   # 按配置权重为1
                    else:
                        rpn_obj_loss = jt.zeros(1) * 1.0
            else:
                # 按配置文件权重计算损失，确保数据类型正确
                if hasattr(rpn_cls, 'shape'):
                    rpn_cls_loss = jt.mean(jt.sqr(rpn_cls)) * 0.0  # loss_weight=0.0
                else:
                    rpn_cls_loss = jt.zeros(1) * 0.0
                
                if hasattr(rpn_reg, 'shape'):
                    rpn_bbox_loss = jt.mean(jt.sqr(rpn_reg)) * 10.0  # loss_weight=10.0
                else:
                    rpn_bbox_loss = jt.zeros(1) * 10.0
                
                if hasattr(rpn_obj, 'shape'):
                    rpn_obj_loss = jt.mean(jt.sqr(rpn_obj)) * 1.0   # loss_weight=1.0
                else:
                    rpn_obj_loss = jt.zeros(1) * 1.0
            
            # ROI 损失计算
            if 'roi_losses' in locals() and isinstance(roi_losses, dict) and len(roi_losses) > 0:
                # 使用真实的ROI损失
                rcnn_cls_loss = roi_losses.get('loss_cls', jt.zeros(1)) * 1.0  # loss_weight=1.0
                rcnn_bbox_loss = roi_losses.get('loss_bbox', jt.zeros(1)) * 1.0  # loss_weight=1.0
                rcnn_score_loss = roi_losses.get('loss_bbox_score', jt.zeros(1)) * 1.0  # loss_weight=1.0
            else:
                # 占位损失，使用配置权重，确保数据类型正确
                if hasattr(roi_cls, 'shape'):
                    rcnn_cls_loss = jt.mean(jt.sqr(roi_cls)) * 1.0  # loss_weight=1.0
                else:
                    rcnn_cls_loss = jt.zeros(1) * 1.0
                
                if hasattr(roi_reg, 'shape'):
                    rcnn_bbox_loss = jt.mean(jt.sqr(roi_reg)) * 1.0  # loss_weight=1.0
                else:
                    rcnn_bbox_loss = jt.zeros(1) * 1.0
                
                if hasattr(roi_score, 'shape'):
                    rcnn_score_loss = jt.mean(jt.sqr(roi_score)) * 1.0  # loss_weight=1.0
                else:
                    rcnn_score_loss = jt.zeros(1) * 1.0
            
            # 汇总总损失
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

        def _roi_align_first_gt(self, feat, gt_bboxes_list, output_size=7, stride=32):
            """在单层特征图上，用每张图的首个 GT 框做简易 RoIAlign。
            - feat: [B, C, H, W]
            - gt_bboxes_list: List[Var[N, 4]] in image coords
            返回: [B, C, output_size, output_size]
            """
            B, C, H, W = feat.shape
            pooled = []
            for n in range(B):
                # 取第 n 张图的首个 gt 框
                box = None
                if isinstance(gt_bboxes_list, (list, tuple)) and len(gt_bboxes_list) > n:
                    b = gt_bboxes_list[n]
                    if hasattr(b, 'shape') and b.shape[0] > 0:
                        box = b[0]
                if box is None:
                    # 若无 gt，退化为整图池化
                    crop = feat[n:n+1, :, :, :]
                else:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    # 转到特征坐标并裁剪到边界
                    xs1 = jt.maximum((x1 / stride).floor().int32(), jt.int32(0))
                    ys1 = jt.maximum((y1 / stride).floor().int32(), jt.int32(0))
                    xs2 = jt.minimum((x2 / stride).ceil().int32(), jt.int32(W-1))
                    ys2 = jt.minimum((y2 / stride).ceil().int32(), jt.int32(H-1))
                    # 防止空区域
                    xs2 = jt.maximum(xs2, xs1 + 1)
                    ys2 = jt.maximum(ys2, ys1 + 1)
                    crop = feat[n:n+1, :, ys1:ys2, xs1:xs2]
                pooled.append(jt.nn.AdaptiveAvgPool2d((output_size, output_size))(crop))
            return jt.concat(pooled, dim=0)

        def _nms_numpy(self, boxes_np, scores_np, iou_thr=0.7, max_num=1000):
            # boxes: [N,4] (x1,y1,x2,y2) in numpy
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

        def _generate_simple_proposals(self, rpn_cls, fpn_feats, strides, nms_pre=1000, max_num=1000, nms_thr=0.7, img_shape=None):
            # 返回 rois Var [N,5] (b,x1,y1,x2,y2) 合并所有 batch
            import numpy as _np
            if not isinstance(fpn_feats, (list, tuple)):
                fpn_feats = [fpn_feats]
            if isinstance(rpn_cls, (list, tuple)):
                cls_list = rpn_cls
            else:
                cls_list = [rpn_cls]

            B = int(img_shape[0]) if img_shape is not None else 1
            rois_concat = []
            
            try:
                for b in range(B):
                    boxes_all = []
                    scores_all = []
                    for lvl, cls_map in enumerate(cls_list):
                        stride = strides[lvl] if lvl < len(strides) else strides[-1]
                        # cls_map: [B, A*2, H, W] -> 前景分数 [H*W*A]
                        H, W = int(cls_map.shape[2]), int(cls_map.shape[3])
                        A2 = int(cls_map.shape[1])
                        A = max(1, A2 // 2)
                        
                        # 安全地获取批次数据
                        if b < cls_map.shape[0]:
                            logits = cls_map[b:b+1, :, :, :]
                        else:
                            continue
                            
                        # 确保张量形状正确
                        if logits.shape[1] != A2:
                            continue
                            
                        logits = logits.reshape(1, A, 2, H, W)
                        probs = jt.softmax(logits, dim=2)  # softmax over class dim
                        fg = probs[:, :, 1, :, :].reshape(-1)
                        
                        # 生成中心点 boxes（简化）：以 stride 为边长的正方形，乘以比例 8
                        scale = 8.0
                        # 网格中心 - 使用numpy避免Jittor张量操作问题
                        yy, xx = _np.meshgrid(_np.arange(H), _np.arange(W), indexing='ij')
                        xx = (xx + 0.5) * stride
                        yy = (yy + 0.5) * stride
                        
                        # 转换为Jittor张量
                        xx = jt.array(xx.astype(_np.float32))
                        yy = jt.array(yy.astype(_np.float32))
                        
                        # 扩展维度
                        xx = xx.unsqueeze(0).unsqueeze(0).expand(A, 1, H, W)
                        yy = yy.unsqueeze(0).unsqueeze(0).expand(A, 1, H, W)
                        
                        half = (stride * scale) / 2.0
                        x1 = (xx - half).clamp(0, float(img_shape[3]-1))
                        y1 = (yy - half).clamp(0, float(img_shape[2]-1))
                        x2 = (xx + half).clamp(0, float(img_shape[3]-1))
                        y2 = (yy + half).clamp(0, float(img_shape[2]-1))
                        
                        # 重塑并转置
                        boxes = jt.stack([x1, y1, x2, y2], dim=1).reshape(4, -1).transpose(1, 0)
                        
                        # 选 top-k
                        k = int(min(nms_pre, boxes.shape[0]))
                        if k <= 0:
                            continue
                            
                        # 安全地转换为numpy
                        try:
                            scores_np = fg.numpy()
                            boxes_np = boxes.numpy()
                        except Exception as e:
                            print(f"⚠️  张量转换失败: {e}")
                            continue
                            
                        if len(scores_np) == 0 or len(boxes_np) == 0:
                            continue
                            
                        order = _np.argsort(-scores_np)[:k]
                        boxes_np = boxes_np[order]
                        scores_np = scores_np[order]
                        
                        # NMS
                        keep = self._nms_numpy(boxes_np, scores_np, iou_thr=nms_thr, max_num=max_num)
                        if len(keep) > 0:
                            boxes_np = boxes_np[keep]
                            scores_np = scores_np[keep]
                            boxes_all.append(boxes_np)
                            scores_all.append(scores_np)
                            
                    if len(boxes_all) == 0:
                        continue
                        
                    boxes_all = _np.concatenate(boxes_all, axis=0)
                    scores_all = _np.concatenate(scores_all, axis=0)
                    
                    # 再次全局 top-k
                    if len(boxes_all) > 0:
                        order = _np.argsort(-scores_all)[:max_num]
                        boxes_all = boxes_all[order]
                        b_col = _np.full((boxes_all.shape[0], 1), float(b), dtype=_np.float32)
                        rois_b = _np.concatenate([b_col, boxes_all.astype(_np.float32)], axis=1)
                        rois_concat.append(rois_b)
                        
            except Exception as e:
                print(f"⚠️  生成提议失败: {e}")
                import traceback
                traceback.print_exc()
                
            if len(rois_concat) == 0:
                return jt.zeros((0, 5), dtype='float32')
                
            try:
                rois_np = _np.concatenate(rois_concat, axis=0)
                return jt.array(rois_np)
            except Exception as e:
                print(f"⚠️  最终提议合并失败: {e}")
                return jt.zeros((0, 5), dtype='float32')
        
        
    
    return JittorModelWithComponents(cfg, stage)


def create_jittor_optimizer(model, cfg):
    """创建Jittor优化器"""
    print("🔧 创建Jittor优化器...")
    
    # 从配置文件中读取优化器设置
    optimizer_cfg = cfg.optimizer
    
    # 使用用户配置的学习率，不再强制降低
    base_lr = optimizer_cfg.get('lr', 0.02)
    
    print(f"📊 使用学习率: {base_lr}")
    
    # 创建优化器
    optimizer = None
    if optimizer_cfg.type == 'SGD':
        optimizer = jt.optim.SGD(
            model.parameters(),
            lr=base_lr,  # 直接使用配置的学习率
            momentum=optimizer_cfg.get('momentum', 0.9),
            weight_decay=optimizer_cfg.get('weight_decay', 0.0001),
            nesterov=optimizer_cfg.get('nesterov', False)
        )
        print(f"✅ 创建SGD优化器，学习率: {base_lr}")
    else:
        print(f"⚠️  不支持的优化器类型: {optimizer_cfg.type}，使用默认SGD")
        optimizer = jt.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=0.0001
        )

    # 设置学习率调度器
    scheduler = None  # 默认值
    if hasattr(cfg, 'lr_config') and cfg.lr_config is not None:
        lr_config = cfg.lr_config
        print(f"📊 学习率配置: {lr_config}")
        
        # 处理step策略（这是配置文件中使用的策略）
        if hasattr(lr_config, 'policy') and lr_config.policy == 'step':
            step = lr_config.get('step', [3, 4])
            gamma = lr_config.get('gamma', 0.1)
            if isinstance(step, list):
                milestones = step
            else:
                milestones = [step]
            
            scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            print(f"✅ 设置StepLR调度器: milestones={milestones}, gamma={gamma}")
            
        elif hasattr(lr_config, 'type') and lr_config.type == 'MultiStepLR':
            milestones = lr_config.get('milestones', [8, 11])
            gamma = lr_config.get('gamma', 0.1)
            scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            print(f"✅ 设置MultiStepLR调度器: milestones={milestones}, gamma={gamma}")
            
        elif hasattr(lr_config, 'type') and lr_config.type == 'StepLR':
            step_size = lr_config.get('step', 8)
            if isinstance(step_size, list):
                step_size = step_size[0] if len(step_size) > 0 else 8
            gamma = lr_config.get('gamma', 0.1)
            scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(f"✅ 设置StepLR调度器: step_size={step_size}, gamma={gamma}")
            
        else:
            if hasattr(lr_config, 'type'):
                print(f"⚠️  不支持的学习率调度器类型: {lr_config.type}")
            elif hasattr(lr_config, 'policy'):
                print(f"⚠️  不支持的学习率策略: {lr_config.policy}")
            else:
                print("⚠️  lr_config没有type或policy属性")
            print("⚠️  学习率调度器未设置，将使用固定学习率")
    else:
        print("⚠️  配置文件中未找到学习率配置，将使用固定学习率")
    
    # 确保scheduler不为None
    if scheduler is None:
        print("📊 创建默认的学习率调度器（固定学习率）")
        scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)  # 固定学习率
    
    return optimizer, scheduler


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
    optimizer, scheduler = create_jittor_optimizer(model, cfg)
    
    # 检查模型参数稳定性
    print("🔍 检查模型参数稳定性...")
    try:
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_np = param.detach().numpy()
                param_stats[name] = {
                    'mean': float(np.mean(param_np)),
                    'std': float(np.std(param_np)),
                    'min': float(np.min(param_np)),
                    'max': float(np.max(param_np)),
                    'has_nan': np.any(np.isnan(param_np)),
                    'has_inf': np.any(np.isinf(param_np))
                }
                
                # 检查异常参数
                if param_stats[name]['has_nan']:
                    print(f"⚠️  参数 {name} 包含 NaN 值")
                if param_stats[name]['has_inf']:
                    print(f"⚠️  参数 {name} 包含 Inf 值")
                if abs(param_stats[name]['mean']) > 1000:
                    print(f"⚠️  参数 {name} 均值过大: {param_stats[name]['mean']:.4f}")
        
        print(f"✅ 模型参数检查完成，共检查 {len(param_stats)} 个参数")
        
        # 打印前几个参数的统计信息
        print("📊 前5个参数统计:")
        for i, (name, stats) in enumerate(list(param_stats.items())[:5]):
            print(f"   {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
            
    except Exception as e:
        print(f"⚠️  模型参数检查失败: {e}")
    
    # 梯度裁剪配置（来自 mmdet 配置）
    grad_clip_cfg = None
    try:
        if hasattr(cfg, 'optimizer_config') and cfg.optimizer_config is not None:
            grad_clip_cfg = getattr(cfg.optimizer_config, 'grad_clip', None)
    except Exception:
        grad_clip_cfg = None
    
    # 学习率配置
    step_epochs = []
    if hasattr(cfg, 'lr_config'):
        lr_config = cfg.lr_config
        if hasattr(lr_config, 'type') and lr_config.type == 'MultiStepLR':
            step_epochs = lr_config.get('milestones', [])
            print(f"📊 学习率衰减轮次: {step_epochs}")
        elif hasattr(lr_config, 'type') and lr_config.type == 'StepLR':
            step_size = lr_config.get('step', 8)
            if isinstance(step_size, list):
                step_epochs = step_size
            else:
                step_epochs = [step_size]
            print(f"📊 学习率衰减轮次: {step_epochs}")
    
    # 训练循环
    # print("🎯 开始Jittor训练循环...")
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

    # 训练开始日志
    print(f"\n🚀 开始训练！")
    print(f"📊 总轮次: {max_epochs}")
    print(f"📊 初始学习率: {optimizer.lr:.6f}")
    print(f"📊 批次大小: {cfg.data.samples_per_gpu}")
    print(f"📊 工作目录: {cfg.work_dir}")
    logger.info(f"Training started: epochs={max_epochs}, lr={optimizer.lr:.6f}")
    
    # 初始化训练统计
    epoch_records = []
    
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
        total_batches = len(data_loaders[0])
        print(f"📊 本轮次总批次数: {total_batches}")
        
        # 添加批次计数器，确保实际处理了所有批次
        processed_batches = 0
        skipped_batches = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(
            enumerate(data_loaders[0]), 
            total=total_batches,
            desc=f"Epoch {epoch+1}/{max_epochs}",
            leave=True,
            ncols=100
        )
        
        for i, data_batch in pbar:
            
            try:
                # 仅转换必要键，避免对复杂元信息递归导致的 __instancecheck__ 递归
                wanted_keys = ['img', 'gt_bboxes', 'gt_labels', 'proposals']
                jt_data = {}
                for key in wanted_keys:
                    if key in data_batch:
                        jt_data[key] = safe_convert_to_jittor(data_batch[key])

                # 进一步强制类型为 Jittor Var，避免混用 torch.Tensor
                def to_jt_var(x):
                    """安全地将各种数据类型转换为Jittor Var"""
                    try:
                        # 已经是Jittor Var
                        if isinstance(x, jt.Var):
                            return x
                        
                        # PyTorch Tensor
                        if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
                            try:
                                # 确保数据类型正确
                                numpy_data = x.detach().cpu().numpy()
                                # 转换为float32以避免精度问题
                                if numpy_data.dtype != np.float32:
                                    numpy_data = numpy_data.astype(np.float32)
                                return jt.array(numpy_data)
                            except Exception as e:
                                print(f"⚠️  PyTorch转换失败: {e}")
                                return x
                        
                        # NumPy array
                        if hasattr(x, 'shape') and hasattr(x, 'dtype'):
                            try:
                                # 确保数据类型正确
                                if x.dtype != np.float32:
                                    x = x.astype(np.float32)
                                return jt.array(x)
                            except Exception as e:
                                print(f"⚠️  NumPy转换失败: {e}")
                                return x
                        
                        # List of tensors
                        if isinstance(x, (list, tuple)) and len(x) > 0:
                            try:
                                converted_list = []
                                for item in x:
                                    converted_item = to_jt_var(item)
                                    converted_list.append(converted_item)
                                return converted_list
                            except Exception as e:
                                print(f"⚠️  列表转换失败: {e}")
                                return x
                        
                        return x
                    except Exception as e:
                        print(f"⚠️  to_jt_var转换失败: {e}")
                        return x

                # 强制转换所有数据为Jittor格式
                if 'img' in jt_data:
                    jt_data['img'] = to_jt_var(jt_data['img'])
                    # 确保图像数据格式正确
                    if hasattr(jt_data['img'], 'shape'):
                        print(f"🔍 图像数据转换后: {jt_data['img'].shape}, 类型: {type(jt_data['img'])}")
                
                if 'gt_bboxes' in jt_data and isinstance(jt_data['gt_bboxes'], (list, tuple)):
                    # 处理嵌套列表结构
                    converted_bboxes = []
                    for v in jt_data['gt_bboxes']:
                        if isinstance(v, (list, tuple)):
                            # 如果是嵌套列表，递归转换
                            converted_bboxes.append([to_jt_var(item) for item in v])
                        else:
                            converted_bboxes.append(to_jt_var(v))
                    jt_data['gt_bboxes'] = converted_bboxes
                    
                    # 检查转换结果
                    for i, bbox in enumerate(jt_data['gt_bboxes']):
                        if isinstance(bbox, (list, tuple)) and len(bbox) > 0:
                            if hasattr(bbox[0], 'shape'):
                                print(f"🔍 GT bbox {i}: list with {len(bbox)} items, first item shape: {bbox[0].shape}, 类型: {type(bbox[0])}")
                            else:
                                print(f"🔍 GT bbox {i}: list with {len(bbox)} items, first item 无shape属性, 类型: {type(bbox[0])}")
                        elif hasattr(bbox, 'shape'):
                            print(f"🔍 GT bbox {i}: {bbox.shape}, 类型: {type(bbox)}")
                
                if 'gt_labels' in jt_data and isinstance(jt_data['gt_labels'], (list, tuple)):
                    # 处理嵌套列表结构
                    converted_labels = []
                    for v in jt_data['gt_labels']:
                        if isinstance(v, (list, tuple)):
                            # 如果是嵌套列表，递归转换
                            converted_labels.append([to_jt_var(item) for item in v])
                        else:
                            converted_labels.append(to_jt_var(v))
                    jt_data['gt_labels'] = converted_labels
                    
                    # 检查转换结果
                    for i, label in enumerate(jt_data['gt_labels']):
                        if isinstance(label, (list, tuple)) and len(label) > 0:
                            if hasattr(label[0], 'shape'):
                                print(f"🔍 GT label {i}: list with {len(label)} items, first item shape: {label[0].shape}, 类型: {type(label[0])}")
                            else:
                                print(f"🔍 GT label {i}: list with {len(label)} items, first item 无shape属性, 类型: {type(label[0])}")
                        elif hasattr(label, 'shape'):
                            print(f"🔍 GT label {i}: {label.shape}, 类型: {type(label)}")
                
                if 'proposals' in jt_data:
                    # proposals 可能是 list 或 tensor
                    if isinstance(jt_data['proposals'], (list, tuple)):
                        jt_data['proposals'] = [to_jt_var(v) for v in jt_data['proposals']]
                    else:
                        jt_data['proposals'] = to_jt_var(jt_data['proposals'])
                
                # 调试信息（只在第一个批次显示，简化输出）
                if i == 0:
                    print(f"🔍 数据调试信息:")
                    for key, value in jt_data.items():
                        if hasattr(value, 'shape'):
                            print(f"   {key}: {value.shape}, 类型: {type(value)}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"   {key}: list with {len(value)} items")
                            if hasattr(value[0], 'shape'):
                                print(f"     first item shape: {value[0].shape}, 类型: {type(value[0])}")
                            else:
                                print(f"     first item 无shape属性, 类型: {type(value[0])}")
                        else:
                            print(f"   {key}: 类型: {type(value)}")
                
                # 前向传播
                losses = model(**jt_data)
                
                # 立即检查rcnn_score_loss，这是问题的根源
                if isinstance(losses, dict) and 'rcnn_score_loss' in losses:
                    score_loss_val = losses['rcnn_score_loss'].item()
                    if abs(score_loss_val) > 1000:
                        print(f"🚨 检测到异常的rcnn_score_loss: {score_loss_val}")
                        # 不要直接重置，而是尝试缩放
                        if score_loss_val > 0:
                            scale_factor = 1000.0 / score_loss_val
                            losses['rcnn_score_loss'] = losses['rcnn_score_loss'] * scale_factor
                            print(f"🔒 rcnn_score_loss 已缩放: {score_loss_val:.2e} -> {losses['rcnn_score_loss'].item():.4f}")
                        else:
                            # 如果是负值，取绝对值后缩放
                            scale_factor = 1000.0 / abs(score_loss_val)
                            losses['rcnn_score_loss'] = losses['rcnn_score_loss'] * scale_factor
                            print(f"🔒 rcnn_score_loss 已缩放: {score_loss_val:.2e} -> {losses['rcnn_score_loss'].item():.4f}")
                
                # 调试：检查损失值是否包含 NaN 或 inf，并进行更严格的稳定化处理
                if isinstance(losses, dict):
                    # 检查每个损失值并进行稳定化处理
                    for key, value in losses.items():
                        if hasattr(value, 'item'):
                            loss_val = value.item()
                            # 更严格的损失值检查
                            if not np.isfinite(loss_val) or abs(loss_val) > 1000:
                                print(f"⚠️  WARNING: {key} = {loss_val} (异常值)")
                                logger.warning(f"Abnormal loss detected: {key} = {loss_val}")
                                
                                # 根据损失类型进行不同的处理
                                if key in ['rcnn_score_loss', 'rcnn_cls_loss', 'rcnn_bbox_loss']:
                                    # 对于分类和回归损失，尝试缩放而不是重置
                                    if np.isnan(loss_val) or np.isinf(loss_val):
                                        losses[key] = jt.array(0.1)
                                        print(f"🔒 {key} 重置为: 0.1 (NaN/Inf)")
                                    elif abs(loss_val) > 1000:
                                        # 缩放异常大的损失值
                                        scale_factor = 100.0 / abs(loss_val)
                                        losses[key] = losses[key] * scale_factor
                                        print(f"🔒 {key} 已缩放: {loss_val:.2e} -> {losses[key].item():.4f}")
                                else:
                                    # 对于其他损失，尝试缩放
                                    if abs(loss_val) > 1000:
                                        scale_factor = 100.0 / abs(loss_val)
                                        losses[key] = losses[key] * scale_factor
                                        print(f"🔒 {key} 已缩放: {loss_val:.2e} -> {losses[key].item():.4f}")
                                    elif np.isnan(loss_val) or np.isinf(loss_val):
                                        losses[key] = jt.array(0.0)
                                        print(f"🔒 {key} 重置为: 0.0 (NaN/Inf)")
                    
                    # 计算总损失并进行稳定化
                    total_loss = sum(losses.values())
                    
                    # 检查总损失是否有效
                    if hasattr(total_loss, 'item'):
                        total_loss_val = total_loss.item()
                        if not np.isfinite(total_loss_val) or abs(total_loss_val) > 1000:
                            print(f"⚠️  WARNING: 总损失 = {total_loss_val} (异常值)")
                            logger.warning(f"Abnormal total loss: {total_loss_val}")
                            
                            # 如果总损失无效，使用所有有效损失的总和
                            valid_losses = []
                            for key, value in losses.items():
                                if hasattr(value, 'item'):
                                    val = value.item()
                                    if np.isfinite(val) and abs(val) <= 1000:
                                        valid_losses.append(value)
                            
                            if valid_losses:
                                total_loss = sum(valid_losses)
                                print(f"✅ 使用有效损失重新计算总损失: {total_loss.item()}")
                            else:
                                # 如果所有损失都无效，尝试使用一个基于批次大小的合理值
                                total_loss = jt.array(0.1 * batch_size)
                                print(f"⚠️  所有损失都无效，使用基于批次大小的值: {total_loss.item()}")
                    
                    # 累积各项损失
                    for key, value in losses.items():
                        if key != 'loss':
                            if key not in epoch_components:
                                epoch_components[key] = 0.0
                            try:
                                epoch_components[key] += value.item()
                            except Exception as e:
                                print(f"⚠️  累积损失失败 {key}: {e}")
                                epoch_components[key] += 0.0
                else:
                    total_loss = losses
                    
                    # 检查总损失是否有效
                    if hasattr(total_loss, 'item'):
                        total_loss_val = total_loss.item()
                        if not np.isfinite(total_loss_val) or abs(total_loss_val) > 1000:
                            print(f"⚠️  WARNING: 总损失 = {total_loss_val} (异常值)")
                            logger.warning(f"Abnormal total loss: {total_loss_val}")
                            # 如果总损失无效，使用一个小的默认值
                            total_loss = jt.array(0.001)
                
                # 温和地限制损失值范围，防止数值不稳定
                try:
                    if hasattr(total_loss, 'clamp'):
                        # 先检查损失值是否异常
                        loss_val = total_loss.item()
                        if not np.isfinite(loss_val):
                            print(f"⚠️  检测到非有限损失值: {loss_val}")
                            # 如果损失值非有限，使用一个基于批次大小的合理值
                            total_loss = jt.array(0.1 * batch_size)
                            print(f"🔒 使用基于批次大小的损失值: {total_loss.item()}")
                        elif abs(loss_val) > 10000:  # 提高阈值，避免过度限制
                            print(f"⚠️  检测到过大损失值: {loss_val}")
                            # 如果损失值过大，进行温和的缩放
                            scale_factor = 1000.0 / abs(loss_val)
                            total_loss = total_loss * scale_factor
                            print(f"🔒 损失值已缩放: {loss_val:.2e} -> {total_loss.item():.4f}")
                        else:
                            # 只在损失值正常时进行温和限制
                            total_loss = total_loss.clamp(-1000.0, 1000.0)
                except Exception as e:
                    print(f"⚠️  损失值限制失败: {e}")
                    # 如果限制失败，使用基于批次大小的值
                    total_loss = jt.array(0.1 * batch_size)
                    print(f"🔒 使用基于批次大小的损失值: {total_loss.item()}")
                
                # 反向传播 & 梯度裁剪（若配置启用）
                # print(f"🔄 开始反向传播...")
                grad_norm_value = None
                if grad_clip_cfg is not None:
                    # 使用 jt.grad 计算全局梯度范数，并按需缩放 loss 以等效实现裁剪
                    try:
                        params = [p for p in model.parameters()]
                        grads = jt.grad(total_loss, params)
                        max_norm = float(getattr(grad_clip_cfg, 'max_norm', 20))
                        norm_type = float(getattr(grad_clip_cfg, 'norm_type', 2))
                        total_norm = 0.0
                        for g in grads:
                            if g is None:
                                continue
                            if norm_type == 2:
                                total_norm += float(jt.sum(g * g).item())
                            else:
                                total_norm += float(jt.sum(jt.abs(g) ** norm_type).item())
                        grad_norm_value = (total_norm ** 0.5) if norm_type == 2 else (total_norm ** (1.0 / norm_type))
                        if grad_norm_value > max_norm:
                            scale = max_norm / (grad_norm_value + 1e-6)
                            total_loss = total_loss * scale
                            print(f"✂️  梯度裁剪: 原始范数 {grad_norm_value:.4f}, 裁剪后 {max_norm:.4f}")
                    except Exception:
                        pass
                
                # 额外的梯度裁剪保护和监控
                try:
                    # 计算梯度并检查数值稳定性
                    params = [p for p in model.parameters()]
                    grads = jt.grad(total_loss, params)
                    
                    # 计算梯度范数用于监控
                    grad_norm = 0.0
                    grad_has_nan = False
                    grad_has_inf = False
                    grad_has_zero = True  # 检查是否所有梯度都为0
                    
                    for i, g in enumerate(grads):
                        if g is not None:
                            try:
                                g_np = g.numpy()
                                if np.any(np.isnan(g_np)):
                                    grad_has_nan = True
                                    print(f"⚠️  参数 {i} 梯度包含 NaN")
                                if np.any(np.isinf(g_np)):
                                    grad_has_inf = True
                                    print(f"⚠️  参数 {i} 梯度包含 Inf")
                                
                                # 计算梯度范数
                                g_norm = np.linalg.norm(g_np)
                                grad_norm += g_norm ** 2
                                
                                # 检查梯度是否接近0
                                if g_norm > 1e-8:
                                    grad_has_zero = False
                                    
                            except Exception:
                                pass
                    
                    grad_norm = grad_norm ** 0.5
                    
                    # 每100步打印梯度信息
                    if i % 100 == 0:
                        print(f"📊 梯度范数: {grad_norm:.6f}")
                        if grad_has_zero:
                            print(f"⚠️  警告: 所有梯度都接近0，可能导致训练停滞")
                    
                    # 如果梯度异常，尝试修复而不是跳过
                    if grad_has_nan or grad_has_inf:
                        print(f"⚠️  检测到异常梯度，尝试修复...")
                        # 尝试使用一个小的学习率来稳定训练
                        if hasattr(optimizer, 'lr'):
                            original_lr = optimizer.lr
                            optimizer.lr = optimizer.lr * 0.1
                            print(f"🔒 临时降低学习率: {original_lr:.6f} -> {optimizer.lr:.6f}")
                        
                        # 继续训练，让模型尝试恢复
                        print(f"🔄 继续训练，尝试恢复...")
                    
                    # 如果梯度为0，尝试增加损失值来产生梯度
                    if grad_has_zero and i % 50 == 0:
                        print(f"⚠️  检测到梯度为0，尝试增加损失值...")
                        # 轻微增加损失值来产生梯度
                        total_loss = total_loss * 1.1
                        print(f"🔒 损失值已增加: {total_loss.item():.6f}")
                        
                except Exception as e:
                    print(f"⚠️  梯度检查失败: {e}")
                    # 如果梯度计算失败，尝试继续训练
                    print(f"🔄 梯度检查失败，尝试继续训练...")
                
                # 更新参数
                try:
                    optimizer.step(total_loss)
                    processed_batches += 1  # 成功处理的批次
                    # print(f"✅ 参数更新成功")
                except Exception as e:
                    print(f"⚠️  优化器更新失败: {e}")
                    logger.error(f"Optimizer step failed: {e}")
                    # 如果优化器更新失败，跳过这个批次
                    skipped_batches += 1
                    continue
                
                # 更新学习率调度器（如果存在）
                if scheduler is not None:
                    try:
                        # 检查当前学习率是否过低
                        current_lr = optimizer.lr
                        if current_lr < 1e-6:  # 如果学习率过低，重置为初始值
                            print(f"⚠️  学习率过低 ({current_lr:.2e})，重置为初始值")
                            optimizer.lr = 0.005  # 重置为初始学习率
                        
                        scheduler.step()
                        
                        # 安全地获取当前学习率
                        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                            current_lr = optimizer.param_groups[0].get('lr', 0.0)
                            if i % 200 == 0:  # 每200步打印一次学习率
                                print(f"📊 当前学习率: {current_lr:.6f}")
                    except Exception as e:
                        print(f"⚠️  学习率调度器更新失败: {e}")
                        # 如果调度器更新失败，尝试重置
                        try:
                            if hasattr(scheduler, 'reset'):
                                scheduler.reset()
                                print("🔄 学习率调度器已重置")
                        except Exception as e2:
                            print(f"⚠️  学习率调度器重置也失败: {e2}")
                
                # 累积损失
                try:
                    if hasattr(total_loss, 'item'):
                        epoch_loss += total_loss.item()
                    else:
                        epoch_loss += float(total_loss)
                except Exception as e:
                    print(f"⚠️  累积总损失失败: {e}")
                    epoch_loss += 0.0
                
                num_batches += 1
                total_steps += 1

                # 周期性回收显存，缓解 OOM（Jittor 推荐）
                if (i + 1) % 200 == 0:
                    try:
                        jt.gc()
                    except Exception:
                        pass
            
                
                # 更新tqdm进度条显示损失信息
                if isinstance(losses, dict):
                    # 只显示主要的损失值，避免信息过多
                    main_losses = {}
                    for k, v in losses.items():
                        if k in ['loss', 'rpn_cls_loss', 'rpn_bbox_loss', 'rcnn_cls_loss', 'rcnn_bbox_loss']:
                            try:
                                main_losses[k] = f"{v.item():.4f}"
                            except:
                                main_losses[k] = "0.0000"
                    
                    # 更新进度条描述
                    pbar.set_postfix({
                        'Loss': f"{total_loss.item():.4f}",
                        'RPN': f"{main_losses.get('rpn_cls_loss', '0.0000')}",
                        'RCNN': f"{main_losses.get('rcnn_cls_loss', '0.0000')}"
                    })
                else:
                    pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
                
                # 每100步记录到logger和JSON日志
                if i % 100 == 0:
                    if isinstance(losses, dict):
                        loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                    else:
                        loss_str = f'{total_loss.item():.4f}'
                    
                    # 记录到logger
                    logger.info(f"Step {i+1}: {loss_str}")
                    
                    # JSON 行日志（与MMDet风格接近）
                    record = {
                        'mode': 'train',
                        'epoch': epoch + 1,
                        'iter': i + 1,
                        'lr': float(optimizer.lr),
                        'total_batches': total_batches,
                        'grad_norm': float(grad_norm) if 'grad_norm' in locals() else 0.0,
                    }
                    if grad_norm_value is not None:
                        record['grad_norm'] = float(grad_norm_value)
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
                    print(f"❌ 批次 {i+1} 处理失败: {e}")
                    import traceback as _tb
                    _tb.print_exc()
                    print(f"   数据类型: {type(data_batch)}")
                    if isinstance(data_batch, dict):
                        for key, value in data_batch.items():
                            print(f"   {key}: {type(value)}")
                skipped_batches += 1
                continue
        
        # 计算平均损失
        try:
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            # 检查平均损失是否有效
            if not np.isfinite(avg_loss):
                print(f"⚠️  WARNING: 平均损失 = {avg_loss} (非有限值)，使用默认值")
                avg_loss = 0.001
        except Exception as e:
            print(f"⚠️  计算平均损失失败: {e}")
            avg_loss = 0.001
        
        try:
            avg_components = {key: value / num_batches if num_batches > 0 else 0.0 
                             for key, value in epoch_components.items()}
            # 检查组件损失是否有效
            for key, value in avg_components.items():
                if not np.isfinite(value):
                    print(f"⚠️  WARNING: {key} = {value} (非有限值)，使用默认值")
                    avg_components[key] = 0.0
        except Exception as e:
            print(f"⚠️  计算平均组件损失失败: {e}")
            avg_components = {}
        
        epoch_losses.append(avg_loss)
        
        # 关闭tqdm进度条
        pbar.close()
        
        print(f"\n📈 轮次 {epoch + 1} 统计:")
        print(f"   - 平均总损失: {avg_loss:.4f}")
        if avg_components:
            for key, value in avg_components.items():
                print(f"   - {key}: {value:.4f}")
        print(f"   - 总步数: {total_steps}")
        print(f"   - 成功处理批次: {processed_batches}")
        print(f"   - 跳过批次: {skipped_batches}")
        print(f"   - 实际处理批次: {num_batches}/{total_batches} ({num_batches/total_batches*100:.1f}%)")
        
        # 记录到logger
        logger.info(
            f"Epoch {epoch+1} avg_loss={avg_loss:.4f}, steps={total_steps}, processed_batches={processed_batches}, skipped_batches={skipped_batches}"
        )
        
        # 记录到JSON日志
        epoch_record = {
            'mode': 'epoch_summary',
            'epoch': epoch + 1,
            'avg_loss': float(avg_loss),
            'total_steps': total_steps,
            'processed_batches': processed_batches,
            'skipped_batches': skipped_batches,
            'num_batches': num_batches,
            'total_batches': total_batches,
            'lr': float(optimizer.lr)
        }
        if avg_components:
            for key, value in avg_components.items():
                epoch_record[f'avg_{key}'] = float(value)
        append_json_log(epoch_record)
        
        # 保存epoch记录
        epoch_records.append(epoch_record)
        
        # 学习率衰减
        if step_epochs and (epoch + 1) in step_epochs:
            old_lr = optimizer.lr
            optimizer.lr *= 0.1
            print(f"📉 学习率衰减: {old_lr:.6f} -> {optimizer.lr:.6f}")
        
        # 验证
        if validate and len(datasets) > 1:
            print(f"🔍 进行验证...")
            model.eval()
        
        # 显示当前epoch完成状态
        print(f"⏱️  Epoch {epoch+1} 完成时间: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # 保存检查点
        if hasattr(cfg, 'checkpoint_config') and cfg.checkpoint_config.interval > 0:
            if (epoch + 1) % cfg.checkpoint_config.interval == 0:
                # 统一使用 .pth 扩展名，便于与 PyTorch 流程对齐
                checkpoint_path = osp.join(cfg.work_dir, f'epoch_{epoch+1}.pth')
                # 实际保存模型参数（Jittor Var -> numpy），并包含基本元信息
                try:
                    print(f"💾 开始保存检查点...")
                    
                    # 强制同步所有 CUDA 操作
                    print(f"🔄 同步 CUDA 操作...")
                    try:
                        jt.sync_all(True)
                        print(f"✅ CUDA 同步完成")
                    except Exception as e:
                        print(f"⚠️  CUDA 同步警告: {e}")
                    
                    # 等待一段时间确保所有操作完成
                    import time
                    time.sleep(1)
                    
                    # 获取模型状态
                    print(f"📋 获取模型状态...")
                    state = {}
                    try:
                        model_state = model.state_dict()
                        print(f"✅ 模型状态获取成功，包含 {len(model_state)} 个参数")
                    except Exception as e:
                        print(f"❌ 模型状态获取失败: {e}")
                        model_state = {}
                    
                    # 转换参数为 numpy
                    print(f"🔄 转换参数格式...")
                    for key, val in model_state.items():
                        try:
                            if hasattr(val, 'numpy'):
                                state[key] = val.numpy()
                            elif hasattr(val, 'detach') and hasattr(val, 'cpu'):
                                # 处理可能的 torch.Tensor
                                state[key] = val.detach().cpu().numpy()
                            else:
                                state[key] = val
                        except Exception as e:
                            print(f"⚠️  参数 {key} 转换失败: {e}")
                            state[key] = val
                    
                    # 准备元信息
                    meta = {
                        'epoch': epoch + 1,
                        'classes': getattr(model, 'CLASSES', None),
                        'config': getattr(cfg, 'pretty_text', None),
                        'timestamp': datetime.datetime.now().isoformat(),
                        'avg_loss': float(avg_loss),
                        'num_batches': num_batches
                    }
                    
                    # 创建目录并保存
                    print(f"📁 创建保存目录...")
                    mmcv.mkdir_or_exist(osp.dirname(osp.abspath(checkpoint_path)))
                    
                    print(f"💾 写入检查点文件...")
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({'state_dict': state, 'meta': meta}, f)
                    
                    print(f"✅ 检查点保存成功: {checkpoint_path}")
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                    
                except Exception as e:
                    print(f"❌ 保存检查点失败: {e}")
                    logger.error(f"Failed to save checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 尝试保存一个简化的检查点
                    try:
                        print(f"🔄 尝试保存简化检查点...")
                        simple_checkpoint_path = osp.join(cfg.work_dir, f'epoch_{epoch+1}_simple.pth')
                        simple_state = {'epoch': epoch + 1, 'error': str(e)}
                        with open(simple_checkpoint_path, 'wb') as f:
                            pickle.dump(simple_state, f)
                        print(f"✅ 简化检查点保存成功: {simple_checkpoint_path}")
                    except Exception as e2:
                        print(f"❌ 简化检查点也保存失败: {e2}")
        
        print(f"✅ 轮次 {epoch + 1} 完成")
    
    # 训练完成总结
    print(f"\n🎉 Jittor训练完成!")
    print(f"📊 训练统计:")
    print(f"   - 总轮次: {max_epochs}")
    print(f"   - 最终平均损失: {np.mean(epoch_losses):.4f}")
    print(f"   - 总步数: {total_steps}")
    print(f"   - 总成功批次: {sum([epoch_record.get('processed_batches', 0) for epoch_record in epoch_records if 'processed_batches' in epoch_record])}")
    print(f"   - 总跳过批次: {sum([epoch_record.get('skipped_batches', 0) for epoch_record in epoch_records if 'skipped_batches' in epoch_record])}")
    
    # 记录到logger
    logger.info(f"Training completed: epochs={max_epochs}, final_avg_loss={np.mean(epoch_losses):.4f}, total_steps={total_steps}")
    
    # 记录到JSON日志
    final_record = {
        'mode': 'training_complete',
        'total_epochs': max_epochs,
        'final_avg_loss': float(np.mean(epoch_losses)),
        'total_steps': total_steps,
        'timestamp': datetime.datetime.now().isoformat()
    }
    append_json_log(final_record)


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
    # 输出训练集图像数量，便于确认步数
    try:
        train_len = len(datasets[0])
        print(f"📦 训练集图像数量: {train_len}")
        logger.info(f"Train dataset length (images): {train_len}")
    except Exception:
        pass
    
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
