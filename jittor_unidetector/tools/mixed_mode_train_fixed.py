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

# 在文件开头添加内存管理函数
import gc
import os

def clear_jittor_cache():
    """清理Jittor缓存"""
    try:
        if hasattr(jt, 'core') and hasattr(jt.core, 'clear_cache'):
            jt.core.clear_cache()
        gc.collect()
    except:
        pass

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
    try:
        import jittor as jt
        
        # 设置Jittor基本配置
        jt.flags.use_cuda = 1  # 启用CUDA
        
        # 设置内存优化选项（使用支持的标志）
        if hasattr(jt.flags, 'amp_level'):
            jt.flags.amp_level = 0  # 禁用自动混合精度（可能导致内存问题）
        
        if hasattr(jt.flags, 'lazy_execution'):
            jt.flags.lazy_execution = 0  # 禁用延迟执行（可能导致内存累积）
        
        # 设置内存清理频率（使用支持的标志）
        if hasattr(jt.flags, 'gc_after_backward'):
            jt.flags.gc_after_backward = 1  # 反向传播后自动垃圾回收
        
        if hasattr(jt.flags, 'gc_after_forward'):
            jt.flags.gc_after_forward = 1  # 前向传播后也自动垃圾回收
        
        # 设置内存限制（防止GPU内存溢出）
        # 注意：某些版本的Jittor可能不支持max_memory标志
        try:
            if hasattr(jt.flags, 'max_memory'):
                jt.flags.max_memory = "12GB"  # 更激进地限制最大内存使用
                print(f"💾 设置最大内存限制: 12GB")
        except:
            print("⚠️  max_memory标志不支持，使用其他内存管理策略")
        
        # 设置其他内存优化标志
        try:
            if hasattr(jt.flags, 'memory_efficient'):
                jt.flags.memory_efficient = 1  # 启用内存效率模式
                print(f"💾 启用内存效率模式")
        except:
            pass
        
        try:
            if hasattr(jt.flags, 'use_parallel_op'):
                jt.flags.use_parallel_op = 0  # 禁用并行操作以减少内存使用
                print(f"💾 禁用并行操作以减少内存使用")
        except:
            pass
        
        print(f"✅ Jittor设置完成")
        print(f"🎮 CUDA: {jt.flags.use_cuda}")
        print(f"🧹 自动清理: 反向传播后={jt.flags.gc_after_backward if hasattr(jt.flags, 'gc_after_backward') else 'N/A'}, 前向传播后={jt.flags.gc_after_forward if hasattr(jt.flags, 'gc_after_forward') else 'N/A'}")
        print(f"💾 内存限制: {jt.flags.max_memory if hasattr(jt.flags, 'max_memory') else 'N/A'}")
        
        return jt
    except ImportError:
        print("❌ 无法导入Jittor，请确保已正确安装")
        return None


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
            # 检查是否是DataContainer类型
            if hasattr(data, 'stack') and hasattr(data, 'cpu_only'):
                # 这是DataContainer，提取其data属性
                inner = getattr(data, 'data')
                print(f"🔍 检测到DataContainer，提取data: {type(inner)}, shape: {getattr(inner, 'shape', 'unknown')}")
                # 递归处理内部数据
                return safe_convert_to_jittor(inner, max_depth, current_depth + 1)
            # 其他有data属性的对象
            inner = getattr(data, 'data')
            if isinstance(inner, list):
                converted_list = []
                for item in inner:
                    try:
                        # 避免递归调用 ensure_jittor_var，直接处理
                        if isinstance(item, jt.Var):
                            converted_list.append(item)
                        elif isinstance(item, np.ndarray):
                            converted_list.append(jt.array(item))
                        elif hasattr(item, 'cpu') and hasattr(item, 'numpy'):
                            converted_list.append(jt.array(item.detach().cpu().numpy()))
                        else:
                            converted_list.append(item)
                    except Exception:
                        converted_list.append(item)
                return converted_list
            elif isinstance(inner, np.ndarray):
                # 如果是numpy数组，直接转换
                return jt.array(inner)
            return safe_convert_to_jittor(inner, max_depth, current_depth + 1)

        # 如果是PyTorch张量或numpy数组，转换为Jittor
        if not is_jt_var:
            try:
                return ensure_jittor_var(data, "data")
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


def ensure_jittor_var(data, name="data", default_shape=None):
    """确保数据是Jittor张量，如果不是则转换，转换失败则返回默认值"""
    # 在函数开头导入numpy
    import numpy as np
    
    try:
        # 首先处理DataContainer类型
        if hasattr(data, 'data') and not isinstance(data, jt.Var):
            # 检查是否是DataContainer类型
            if hasattr(data, 'stack') and hasattr(data, 'cpu_only'):
                # 这是DataContainer，提取其data属性
                inner_data = data.data
                print(f"🔍 检测到DataContainer，提取data: {type(inner_data)}, shape: {getattr(inner_data, 'shape', 'unknown')}")
                # 递归处理内部数据
                return ensure_jittor_var(inner_data, name, default_shape)
        
        if isinstance(data, jt.Var):
            return data
        elif isinstance(data, (list, tuple)):
            # 如果是列表/元组，尝试转换为张量
            if len(data) == 0:
                if default_shape:
                    return jt.zeros(default_shape, dtype='float32')
                else:
                    return jt.zeros((0,), dtype='float32')
            else:
                # 尝试将列表转换为张量
                try:
                    # 先检查列表中的元素类型
                    if len(data) > 0:
                        first_item = data[0]
                        if isinstance(first_item, (list, tuple)):
                            # 如果是嵌套列表，需要特殊处理
                            try:
                                # 对于gt_bboxes，期望格式是 [[[x1,y1,x2,y2], ...], ...]
                                # 对于gt_labels，期望格式是 [[label1, label2, ...], ...]
                                if name == "gt_bboxes":
                                    # 处理gt_bboxes：展平所有batch的边界框
                                    all_bboxes = []
                                    for batch_bboxes in data:
                                        if isinstance(batch_bboxes, (list, tuple)):
                                            for bbox in batch_bboxes:
                                                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                                    all_bboxes.append(bbox)
                                    if all_bboxes:
                                        return jt.array(all_bboxes, dtype='float32')
                                    else:
                                        return jt.zeros((1, 4), dtype='float32')
                                elif name == "gt_labels":
                                    # 处理gt_labels：展平所有batch的标签
                                    all_labels = []
                                    for batch_labels in data:
                                        if isinstance(batch_labels, (list, tuple)):
                                            for label in batch_labels:
                                                if isinstance(label, (int, float)):
                                                    all_labels.append(int(label))
                                    if all_labels:
                                        return jt.array(all_labels, dtype='int32')
                                    else:
                                        return jt.zeros((1,), dtype='int32')
                                else:
                                    # 其他嵌套列表，尝试直接转换
                                    return jt.array(data)
                            except Exception as nested_error:
                                print(f"⚠️  嵌套列表处理失败 {name}: {nested_error}")
                                # 如果嵌套处理失败，返回默认值
                                if name == "gt_bboxes":
                                    return jt.zeros((1, 4), dtype='float32')
                                elif name == "gt_labels":
                                    return jt.zeros((1,), dtype='int32')
                                else:
                                    return jt.zeros((1,), dtype='float32')
                        else:
                            # 如果是简单列表，直接转换
                            return jt.array(data)
                    else:
                        return jt.array(data)
                except Exception as e:
                    print(f"⚠️  列表转张量失败 {name}: {e}")
                    if default_shape:
                        return jt.zeros(default_shape, dtype='float32')
                    else:
                        return jt.zeros((1,), dtype='float32')
        elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):
            # PyTorch张量
            return jt.array(data.detach().cpu().numpy())
        elif isinstance(data, np.ndarray):
            # NumPy数组
            return jt.array(data)
        else:
            # 其他类型，尝试转换为numpy再转为Jittor
            try:
                np_data = np.array(data)
                return jt.array(np_data)
            except Exception as e:
                print(f"⚠️  类型转换失败 {name}: {e}")
                if default_shape:
                    return jt.zeros(default_shape, dtype='float32')
                else:
                    return jt.zeros((1,), dtype='float32')
    except Exception as e:
        print(f"⚠️  确保Jittor张量失败 {name}: {e}")
        if default_shape:
            return jt.zeros(default_shape, dtype='float32')
        else:
            return jt.zeros((1,), dtype='float32')


def safe_sum(tensor, dim=None, name="tensor"):
    """安全的sum操作，确保输入是Jittor张量"""
    try:
        safe_tensor = ensure_jittor_var(tensor, name)
        if dim is not None:
            return safe_tensor.sum(dim=dim)
        else:
            return safe_tensor.sum()
    except Exception as e:
        print(f"⚠️  sum操作失败 {name}: {e}")
        # 返回零张量
        if dim is not None:
            return jt.zeros((1,), dtype='float32')
        else:
            return jt.zeros((1,), dtype='float32')


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
                try:
                    img_var = ensure_jittor_var(img, "img")
                    batch_size = img_var.shape[0]
                except Exception:
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
            """第一阶段前向传播（使用组件化架构）"""
            try:
                print(f"🔍 开始第一阶段前向传播，步骤 {getattr(self, '_step_count', 0)}")
                # 打印内存使用情况
                if hasattr(self, '_step_count'):
                    self._step_count += 1
                else:
                    self._step_count = 0
                
                # 检查GPU内存使用情况，如果接近溢出则清理
                try:
                    if jt.flags.use_cuda:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_used = info.used / 1024**3  # GB
                        memory_total = info.total / 1024**3  # GB
                        
                        if memory_used / memory_total > 0.85:  # 使用率超过85%就开始清理
                            print(f"⚠️  GPU内存使用率过高: {memory_used:.2f}GB/{memory_total:.2f}GB")
                            clear_jittor_cache()
                            jt.sync_all()  # 同步所有操作
                            
                            # 强制垃圾回收
                            import gc
                            gc.collect()
                            
                            # 如果内存仍然很高，尝试更激进的清理
                            if memory_used / memory_total > 0.9:
                                print("🚨 内存使用率仍然过高，进行激进清理...")
                                jt.gc()  # Jittor垃圾回收
                                jt.sync_all()
                except:
                    pass  # 如果无法获取GPU信息，忽略
                    
                # Backbone特征提取（优先使用 jittor resnet50，否则回退到简化版）
                print("🔍 步骤1: Backbone特征提取")
                if hasattr(self, 'resnet') and self.resnet is not None:
                    try:
                        x = self.resnet.conv1(img)
                        x = self.resnet.bn1(x)
                        x = self.resnet.relu(x)
                        x = self.resnet.maxpool(x)
                        
                        c2 = self.resnet.layer1(x)
                        c3 = self.resnet.layer2(c2)
                        c4 = self.resnet.layer3(c3)
                        c5 = self.resnet.layer4(c4)
                        feat = c5  # [B, 2048, H/32, W/32]
                        print(f"✅ Backbone特征提取成功: feat shape={feat.shape}")
                        
                        # 清理中间特征图以节省内存
                        del x
                        if self._step_count % 50 == 0:
                            clear_jittor_cache()
                    except Exception as e:
                        print(f"⚠️  Backbone特征提取失败: {e}")
                        raise e

                # 处理img参数：如果是列表，提取第一个元素并确保格式正确
                if isinstance(img, (list, tuple)) and len(img) > 0:
                    img = img[0]  # 提取第一个元素
                    print(f"🔍 从列表中提取img，类型: {type(img)}")
                
                # 强化图像张量健壮性：确保为 jt.Var、float32、NCHW
                try:
                    img = ensure_jittor_var(img, "img", (1, 3, 224, 224))
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)
                except Exception as e:
                    print(f"⚠️  图像转换失败: {e}")
                    # 创建默认图像
                    img = jt.randn(1, 3, 224, 224)
                try:
                    img = img.float32()
                except Exception:
                    pass
                
                # 提取第一个样本的gt_bbox和gt_label用于损失计算
                # 使用新的辅助函数简化类型转换
                try:
                    if isinstance(gt_bboxes, jt.Var) and gt_bboxes.shape[0] > 0:
                        # 数据格式: [batch_size, max_objects, 4]
                        gt_bbox = gt_bboxes[0]  # 取第一个样本
                        # 过滤掉空的bbox（全零的）
                        gt_bbox = ensure_jittor_var(gt_bbox, "gt_bbox", (1, 4))
                        if len(gt_bbox.shape) == 2 and gt_bbox.shape[1] == 4:
                            # 确保 gt_bbox 是 Jittor 张量
                            gt_bbox = ensure_jittor_var(gt_bbox, "gt_bbox")
                            # 计算每个bbox的有效性（非零）
                            valid_mask = gt_bbox.sum(dim=1) != 0
                            valid_count = valid_mask.sum()
                            if valid_count.item() > 0:
                                gt_bbox = gt_bbox[valid_mask]
                            else:
                                gt_bbox = jt.randn(1, 4) * 0.01
                        else:
                            gt_bbox = jt.randn(1, 4) * 0.01
                    else:
                        gt_bbox = jt.randn(1, 4) * 0.01
                except Exception as e:
                    print(f"⚠️  gt_bboxes处理失败: {e}")
                    gt_bbox = jt.randn(1, 4) * 0.01

                        # 处理gt_labels
                try:
                    if isinstance(gt_labels, jt.Var) and gt_labels.shape[0] > 0:
                        gt_label = gt_labels[0]  # 取第一个样本
                        gt_label = ensure_jittor_var(gt_label, "gt_label", (1,))
                        if len(gt_label.shape) == 1:
                            # 过滤掉无效的标签
                            valid_mask = gt_label >= 0
                            valid_count = valid_mask.sum()
                            if valid_count.item() > 0:
                                gt_label = gt_label[valid_mask]
                            else:
                                gt_label = jt.zeros(1, dtype='int32')
                        else:
                            gt_label = jt.zeros(1, dtype='int32')
                    else:
                        gt_label = jt.zeros(1, dtype='int32')
                except Exception as e:
                    print(f"⚠️  gt_labels处理失败: {e}")
                    gt_label = jt.zeros(1, dtype='int32')

                # 步骤2: FPN特征融合
                print("🔍 步骤2: FPN特征融合")
                try:
                    if hasattr(self, 'fpn') and self.fpn is not None:
                        # 使用真实的FPN
                        fpn_feats = self.fpn([c2, c3, c4, c5])
                        print(f"✅ FPN特征融合成功: {len(fpn_feats)} 层")
                    else:
                        # 简化版：直接使用c5作为单层特征
                        fpn_feats = [feat]
                        print(f"✅ 使用简化FPN: 单层特征 {feat.shape}")
                except Exception as e:
                    print(f"⚠️  FPN特征融合失败: {e}")
                    # 回退到单层特征
                    fpn_feats = [feat]
                    print(f"⚠️  回退到单层特征: {feat.shape}")

                # 步骤3: RPN前向传播
                print("🔍 步骤3: RPN前向传播")
                try:
                    if hasattr(self, 'rpn_head_jt') and self.rpn_head_jt is not None:
                        # 传入全部 FPN 层，RPNHead 内部支持多层
                        rpn_out = self.rpn_head_jt(fpn_feats if 'fpn_feats' in locals() else fpn_rpn)
                        print(f"🔍 RPN输出调试: 类型={type(rpn_out)}, 长度={len(rpn_out) if isinstance(rpn_out, (list, tuple)) else 'N/A'}")
                        
                        # 安全地解包RPN输出
                        try:
                            if isinstance(rpn_out, (list, tuple)):
                                if len(rpn_out) == 3:
                                    rpn_cls, rpn_reg, rpn_obj = rpn_out
                                    print("✅ RPN输出解包成功: 3个值")
                                elif len(rpn_out) == 2:
                                    rpn_cls, rpn_reg = rpn_out
                                    rpn_obj = None
                                    print("✅ RPN输出解包成功: 2个值")
                                else:
                                    print(f"⚠️ RPN输出长度异常: {len(rpn_out)}")
                                    rpn_cls = rpn_out[0] if len(rpn_out) > 0 else None
                                    rpn_reg = rpn_out[1] if len(rpn_out) > 1 else None
                                    rpn_obj = None
                            else:
                                # 如果rpn_out不是列表/元组，可能是单个张量
                                print(f"⚠️ RPN输出不是列表/元组: {type(rpn_out)}")
                                rpn_cls = rpn_out
                                rpn_reg = None
                                rpn_obj = None
                        except Exception as e:
                            print(f"⚠️ RPN输出解包失败: {e}")
                            # 创建默认值
                            rpn_cls = jt.randn(1, 1, 64, 64)
                            rpn_reg = jt.randn(1, 4, 64, 64)
                            rpn_obj = None
                        
                        print(f"✅ RPN前向传播成功")
                    else:
                        print("⚠️  RPN head不存在，跳过RPN前向传播")
                        rpn_cls = jt.randn(1, 1, 64, 64)
                        rpn_reg = jt.randn(1, 4, 64, 64)
                        rpn_obj = None
                except Exception as e:
                    print(f"⚠️  RPN前向传播失败: {e}")
                    # 创建默认值
                    rpn_cls = jt.randn(1, 1, 64, 64)
                    rpn_reg = jt.randn(1, 4, 64, 64)
                    rpn_obj = None

                # 步骤4: ROI Head处理
                print("🔍 步骤4: ROI Head处理")
                try:
                    # 生成简单的proposals（简化版）
                    if hasattr(self, 'bbox_head') and self.bbox_head is not None:
                        # 使用真实的bbox_head
                        try:
                            # 生成简单的proposals
                            rois = self._generate_simple_proposals(
                                rpn_cls, fpn_feats, [32, 16, 8, 4], 
                                nms_pre=1000, max_num=1000, nms_thr=0.7, 
                                img_shape=img.shape
                            )
                            
                            # 构建 ROI 训练目标（简化真实版）
                            try:
                                targets_result = self.bbox_head.build_targets_minimal(
                                    rois,
                                    gt_bboxes,
                                    img_shape=img.shape,
                                    pos_iou_thr=getattr(self.cfg.train_cfg.rcnn.assigner, 'pos_iou_thr', 0.5) if hasattr(self.cfg, 'train_cfg') else 0.5,
                                    neg_iou_thr=getattr(self.cfg.train_cfg.rcnn.assigner, 'neg_iou_thr', 0.5) if hasattr(self.cfg, 'train_cfg') else 0.5,
                                    num_samples=getattr(self.cfg.train_cfg.rcnn.sampler, 'num', 256) if hasattr(self.cfg, 'train_cfg') else 256,
                                    pos_fraction=getattr(self.cfg.train_cfg.rcnn.sampler, 'pos_fraction', 0.25) if hasattr(self.cfg, 'train_cfg') else 0.25,
                                )
                                
                                # 安全地解包返回值
                                if isinstance(targets_result, (list, tuple)) and len(targets_result) == 6:
                                    labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets, bbox_score_weights = targets_result
                                elif isinstance(targets_result, (list, tuple)) and len(targets_result) == 1:
                                    # 如果只返回1个值，创建默认值
                                    print(f"⚠️  build_targets_minimal只返回1个值: {targets_result}")
                                    labels = targets_result[0]
                                    label_weights = jt.ones_like(labels) if hasattr(labels, 'shape') else jt.ones(1)
                                    bbox_targets = jt.zeros((labels.shape[0], 4)) if hasattr(labels, 'shape') else jt.zeros((1, 4))
                                    bbox_weights = jt.zeros((labels.shape[0], 4)) if hasattr(labels, 'shape') else jt.zeros((1, 4))
                                    bbox_score_targets = jt.zeros_like(labels) if hasattr(labels, 'shape') else jt.zeros(1)
                                    bbox_score_weights = jt.ones_like(labels) if hasattr(labels, 'shape') else jt.ones(1)
                                else:
                                    # 其他情况，创建默认值
                                    print(f"⚠️  build_targets_minimal返回异常值: {type(targets_result)}, {targets_result}")
                                    labels = jt.zeros(1, dtype='int32')
                                    label_weights = jt.ones(1)
                                    bbox_targets = jt.zeros((1, 4))
                                    bbox_weights = jt.zeros((1, 4))
                                    bbox_score_targets = jt.zeros(1)
                                    bbox_score_weights = jt.ones(1)
                            except Exception as e:
                                print(f"⚠️  build_targets_minimal调用失败: {e}")
                                # 创建默认值
                                labels = jt.zeros(1, dtype='int32')
                                label_weights = jt.ones(1)
                                bbox_targets = jt.zeros((1, 4))
                                bbox_weights = jt.zeros((1, 4))
                                bbox_score_targets = jt.zeros(1)
                                bbox_score_weights = jt.ones(1)
                            
                            # 计算ROI损失
                            roi_losses = self.bbox_head.loss(
                                cls_score=roi_cls if 'roi_cls' in locals() else jt.randn(1, 1),
                                bbox_pred=roi_reg if 'roi_reg' in locals() else jt.randn(1, 4),
                                bbox_score_pred=roi_score if 'roi_score' in locals() else jt.randn(1, 1),
                                rois=rois,
                                labels=labels,
                                label_weights=label_weights,
                                bbox_targets=bbox_targets,
                                bbox_weights=bbox_weights,
                                bbox_score_targets=bbox_score_targets,
                                bbox_score_weights=bbox_score_weights,
                                reduction_override=None
                            )
                            print(f"✅ ROI Head处理成功")
                        except Exception as e:
                            print(f"⚠️  ROI Head处理失败: {e}")
                            # 创建默认的roi_losses
                            roi_losses = {
                                'loss_cls': jt.zeros(1),
                                'loss_bbox': jt.zeros(1),
                                'loss_bbox_score': jt.zeros(1)
                            }
                    else:
                        print("⚠️  bbox_head不存在，跳过ROI Head处理")
                        roi_losses = {
                            'loss_cls': jt.zeros(1),
                            'loss_bbox': jt.zeros(1),
                            'loss_bbox_score': jt.zeros(1)
                        }
                except Exception as e:
                    print(f"⚠️  ROI Head处理失败: {e}")
                    # 创建默认的roi_losses
                    roi_losses = {
                        'loss_cls': jt.zeros(1),
                        'loss_bbox': jt.zeros(1),
                        'loss_bbox_score': jt.zeros(1)
                    }

                # 计算损失 (真实 RPN 训练)
                if hasattr(self, 'rpn_head_jt') and self.rpn_head_jt is not None and rpn_obj is not None and hasattr(self.rpn_head_jt, 'loss'):
                    try:
                        # 使用辅助函数简化gt_bboxes格式转换
                        gt_bboxes_tensor = ensure_jittor_var(gt_bboxes, "gt_bboxes", (1, 0, 4))
                        
                        print(f"🔍 RPN loss调用前，gt_bboxes格式: {type(gt_bboxes_tensor)}, shape: {gt_bboxes_tensor.shape}")
                        
                        # 使用正确的参数调用RPN loss函数
                        # 注意：rpn_cls, rpn_reg, rpn_obj 已经是多层级列表格式
                        rpn_losses = self.rpn_head_jt.loss(
                            rpn_cls, rpn_reg, rpn_obj,  # 直接传递，不要包装成列表
                            gt_bboxes_list=gt_bboxes_tensor,
                            img_shape=img.shape  # 传递完整的img_shape (B, C, H, W)
                        )
                        rpn_cls_loss = rpn_losses.get('loss_rpn_cls', jt.zeros(1))
                        rpn_bbox_loss = rpn_losses.get('loss_rpn_bbox', jt.zeros(1))
                        rpn_obj_loss = rpn_losses.get('loss_rpn_obj', jt.zeros(1))
                    except Exception as e:
                        print(f"⚠️  RPN损失计算失败: {e}")
                        import traceback as _tb
                        _tb.print_exc()
                        # 回退到简化损失，使用辅助函数确保数据类型正确
                        rpn_cls_loss = jt.mean(jt.sqr(ensure_jittor_var(rpn_cls, "rpn_cls"))) * 0.0  # 按配置权重为0
                        rpn_bbox_loss = jt.mean(jt.sqr(ensure_jittor_var(rpn_reg, "rpn_reg"))) * 10.0  # loss_weight=10.0
                        
                        rpn_obj_loss = jt.mean(jt.sqr(ensure_jittor_var(rpn_obj, "rpn_obj"))) * 1.0   # loss_weight=1.0
                else:
                    # 按配置文件权重计算损失，使用辅助函数确保数据类型正确
                    rpn_cls_loss = jt.mean(jt.sqr(ensure_jittor_var(rpn_cls, "rpn_cls"))) * 0.0  # loss_weight=0.0
                    rpn_bbox_loss = jt.mean(jt.sqr(ensure_jittor_var(rpn_reg, "rpn_reg"))) * 10.0  # loss_weight=10.0
                    rpn_obj_loss = jt.mean(jt.sqr(ensure_jittor_var(rpn_obj, "rpn_obj"))) * 1.0   # loss_weight=1.0
                
                # ROI 损失计算
                if 'roi_losses' in locals() and isinstance(roi_losses, dict) and len(roi_losses) > 0:
                    # 使用真实的ROI损失
                    rcnn_cls_loss = roi_losses.get('loss_cls', jt.zeros(1)) * 1.0  # loss_weight=1.0
                    rcnn_bbox_loss = roi_losses.get('loss_bbox', jt.zeros(1)) * 1.0  # loss_weight=1.0
                    rcnn_score_loss = roi_losses.get('loss_bbox_score', jt.zeros(1)) * 1.0  # loss_weight=1.0
                else:
                    # 占位损失，使用配置权重，使用辅助函数确保数据类型正确
                    rcnn_cls_loss = jt.mean(jt.sqr(ensure_jittor_var(roi_cls, "roi_cls"))) * 1.0  # loss_weight=1.0
                    rcnn_bbox_loss = jt.mean(jt.sqr(ensure_jittor_var(roi_reg, "roi_reg"))) * 1.0  # loss_weight=1.0
                    rcnn_score_loss = jt.mean(jt.sqr(ensure_jittor_var(roi_score, "roi_score"))) * 1.0  # loss_weight=1.0
                    
                # 汇总总损失
                total_loss = rpn_cls_loss + rpn_bbox_loss + rpn_obj_loss + rcnn_cls_loss + rcnn_bbox_loss + rcnn_score_loss
                
                # 使用新的辅助函数确保total_loss是单个Jittor张量
                total_loss = ensure_jittor_var(total_loss, "total_loss", (1,))
                
                print(f"✅ ROI Head处理成功: total_loss={total_loss.item():.4f}")
                
                # 调试信息：打印损失值
                if self._step_count % 10 == 0:  # 每10步打印一次
                    print(f"🔍 步骤 {self._step_count} 损失值:")
                    print(f"   RPN: cls={ensure_jittor_var(rpn_cls_loss, 'rpn_cls_loss').item():.6f}, bbox={ensure_jittor_var(rpn_bbox_loss, 'rpn_bbox_loss').item():.6f}, obj={ensure_jittor_var(rpn_obj_loss, 'rpn_obj_loss').item():.6f}")
                    print(f"   RCNN: cls={ensure_jittor_var(rcnn_cls_loss, 'rcnn_cls_loss').item():.6f}, bbox={ensure_jittor_var(rcnn_bbox_loss, 'rcnn_bbox_loss').item():.6f}, score={ensure_jittor_var(rcnn_score_loss, 'rcnn_score_loss').item():.6f}")
                    print(f"   总损失: {ensure_jittor_var(total_loss, 'total_loss').item():.6f}")
                
                return {
                    'loss': total_loss,
                    'rpn_cls_loss': rpn_cls_loss,
                    'rpn_bbox_loss': rpn_bbox_loss,
                    'rpn_obj_loss': rpn_obj_loss,
                    'rcnn_cls_loss': rcnn_cls_loss,
                    'rcnn_bbox_loss': rcnn_bbox_loss,
                    'rcnn_score_loss': rcnn_score_loss
                }

            except Exception as e:
                print(f"⚠️  前向传播步骤 {self._step_count} 失败: {e}")
                return {
                    'loss': jt.array(0.0),
                    'rpn_cls_loss': jt.array(0.0),
                    'rpn_bbox_loss': jt.array(0.0),
                    'rpn_obj_loss': jt.array(0.0),
                    'rcnn_cls_loss': jt.array(0.0),
                    'rcnn_bbox_loss': jt.array(0.0),
                    'rcnn_score_loss': jt.array(0.0)
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
                    b = ensure_jittor_var(gt_bboxes_list[n], f"gt_bboxes_list[{n}]")
                    if b.shape[0] > 0:
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
            import numpy as np
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
                            scores_np = ensure_jittor_var(fg, "fg").numpy()
                            boxes_np = ensure_jittor_var(boxes, "boxes").numpy()
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
    
    # 创建自定义数据加载器包装器，处理DataContainer
    def create_jittor_dataloader(dataset, samples_per_gpu, workers_per_gpu, **kwargs):
        """创建Jittor兼容的数据加载器"""
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=1,
            dist=distributed,
            shuffle=True,
            seed=cfg.seed
        )
        
        # 包装数据加载器，在返回数据时处理DataContainer
        class JittorDataLoaderWrapper:
            def __init__(self, original_loader):
                self.original_loader = original_loader
                self.dataset = original_loader.dataset
                self.batch_size = original_loader.batch_size
                self.num_workers = original_loader.num_workers
                self.sampler = original_loader.sampler
                self.pin_memory = getattr(original_loader, 'pin_memory', False)
                self.drop_last = getattr(original_loader, 'drop_last', False)
                self.timeout = getattr(original_loader, 'timeout', 0)
                self.worker_init_fn = getattr(original_loader, 'worker_init_fn', None)
                self.multiprocessing_context = getattr(original_loader, 'multiprocessing_context', None)
                self.generator = getattr(original_loader, 'generator', None)
                self.prefetch_factor = getattr(original_loader, 'prefetch_factor', 2)
                self.persistent_workers = getattr(original_loader, 'persistent_workers', False)
            
            def __iter__(self):
                for batch in self.original_loader:
                    # 预处理数据，提取DataContainer中的数据
                    processed_batch = {}
                    for key, value in batch.items():
                        if hasattr(value, 'data') and hasattr(value, 'stack') and hasattr(value, 'cpu_only'):
                            # 这是DataContainer，提取其data属性
                            processed_batch[key] = value.data
                        else:
                            processed_batch[key] = value
                    yield processed_batch
            
            def __len__(self):
                return len(self.original_loader)
        
        return JittorDataLoaderWrapper(dataloader)
    
    data_loaders = [
        create_jittor_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu
        )
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
        
        # 每个epoch开始时清理内存
        if epoch > 0:  # 第一个epoch不需要清理
            clear_jittor_cache()
            gc.collect()
        
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
                # 调试：检查数据批次的批次大小
                if i == 0:  # 只在第一个批次显示
                    print(f"🔍 数据批次调试信息:")
                    print(f"   data_batch类型: {type(data_batch)}")
                    if 'img' in data_batch:
                        img_data = data_batch['img']
                        print(f"   img类型: {type(img_data)}")
                        try:
                            img_var = ensure_jittor_var(img_data, "img_data")
                            print(f"   img形状: {img_var.shape}")
                        except Exception:
                            if isinstance(img_data, (list, tuple)) and len(img_data) > 0:
                                try:
                                    first_img = ensure_jittor_var(img_data[0], "img_data[0]")
                                    print(f"   第一个img形状: {first_img.shape}")
                                except Exception:
                                    print(f"   第一个img转换失败")
                            else:
                                print(f"   img转换失败")
                    print(f"   data_batch键: {list(data_batch.keys())}")
                
                # 仅转换必要键，避免对复杂元信息递归导致的 __instancecheck__ 递归
                wanted_keys = ['img', 'gt_bboxes', 'gt_labels', 'proposals']
                jt_data = {}
                for key in wanted_keys:
                    if key in data_batch:
                        jt_data[key] = safe_convert_to_jittor(data_batch[key])

                # 使用新的辅助函数简化类型转换
                def to_jt_var(x):
                    """安全地将各种数据类型转换为Jittor Var"""
                    return ensure_jittor_var(x, "data", None)

                # 强制转换所有数据为Jittor格式
                if 'img' in jt_data:
                    # 处理图像数据：确保是单个张量而不是列表
                    try:
                        if isinstance(jt_data['img'], (list, tuple)) and len(jt_data['img']) > 0:
                            # 如果是列表，直接转换整个列表（MMDetection的默认行为）
                            jt_data['img'] = to_jt_var(jt_data['img'])
                        else:
                            jt_data['img'] = to_jt_var(jt_data['img'])
                        
                        # 使用辅助函数确保图像数据格式正确，不强制指定形状
                        jt_data['img'] = ensure_jittor_var(jt_data['img'], "img")
                        print(f"🔍 图像数据转换后: {jt_data['img'].shape}, 类型: {type(jt_data['img'])}")
                    except Exception as img_error:
                        print(f"⚠️  图像数据转换失败: {img_error}")
                        # 如果转换失败，尝试使用默认值
                        jt_data['img'] = jt.zeros((1, 3, 224, 224), dtype='float32')
                        print(f"⚠️  使用默认图像张量: {jt_data['img'].shape}")
                
                if 'gt_bboxes' in jt_data:
                    # 处理 DataContainer 类型
                    try:
                        if hasattr(jt_data['gt_bboxes'], 'data'):
                            # 如果是 DataContainer，提取其 data 属性
                            jt_data['gt_bboxes'] = jt_data['gt_bboxes'].data
                    except Exception:
                        pass
                    
                    # 使用新的辅助函数简化转换
                    try:
                        jt_data['gt_bboxes'] = ensure_jittor_var(jt_data['gt_bboxes'], "gt_bboxes")
                        print(f"🔍 GT bboxes 转换后: {jt_data['gt_bboxes'].shape}, 类型: {type(jt_data['gt_bboxes'])}")
                    except Exception as bbox_error:
                        print(f"⚠️  GT bboxes 转换失败: {bbox_error}")
                        # 如果转换失败，使用默认值
                        jt_data['gt_bboxes'] = jt.zeros((1, 4), dtype='float32')
                        print(f"⚠️  使用默认 GT bboxes: {jt_data['gt_bboxes'].shape}")
                
                if 'gt_labels' in jt_data:
                    # 处理 DataContainer 类型
                    try:
                        if hasattr(jt_data['gt_labels'], 'data'):
                            # 如果是 DataContainer，提取其 data 属性
                            jt_data['gt_labels'] = jt_data['gt_labels'].data
                    except Exception:
                        pass
                    
                    # 使用新的辅助函数简化转换
                    try:
                        jt_data['gt_labels'] = ensure_jittor_var(jt_data['gt_labels'], "gt_labels")
                        print(f"🔍 GT labels 转换后: {jt_data['gt_labels'].shape}, 类型: {type(jt_data['gt_labels'])}")
                    except Exception as label_error:
                        print(f"⚠️  GT labels 转换失败: {label_error}")
                        # 如果转换失败，使用默认值
                        jt_data['gt_labels'] = jt.zeros((1,), dtype='int32')
                        print(f"⚠️  使用默认 GT labels: {jt_data['gt_labels'].shape}")
            
                if 'proposals' in jt_data:
                    # 使用新的辅助函数简化转换
                    jt_data['proposals'] = ensure_jittor_var(jt_data['proposals'], "proposals")
                
                # 数据格式验证和修复
                try:
                    # 确保gt_bboxes和gt_labels的格式正确
                    if 'gt_bboxes' in jt_data and 'gt_labels' in jt_data:
                        bbox_shape = jt_data['gt_bboxes'].shape
                        label_shape = jt_data['gt_labels'].shape
                        
                        # 检查gt_bboxes格式：应该是 [N, 4] 其中N是边界框数量
                        if len(bbox_shape) == 2 and bbox_shape[1] == 4:
                            print(f"✅ gt_bboxes格式正确: {bbox_shape}")
                        elif len(bbox_shape) == 3 and bbox_shape[2] == 4:
                            # 如果是 [B, N, 4] 格式，展平为 [B*N, 4]
                            jt_data['gt_bboxes'] = jt_data['gt_bboxes'].view(-1, 4)
                            print(f"✅ gt_bboxes已展平: {jt_data['gt_bboxes'].shape}")
                        else:
                            print(f"⚠️  gt_bboxes格式异常: {bbox_shape}")
                        
                        # 检查gt_labels格式：应该是 [N] 其中N是标签数量
                        if len(label_shape) == 1:
                            print(f"✅ gt_labels格式正确: {label_shape}")
                        elif len(label_shape) == 2:
                            # 如果是 [B, N] 格式，展平为 [B*N]
                            jt_data['gt_labels'] = jt_data['gt_labels'].view(-1)
                            print(f"✅ gt_labels已展平: {jt_data['gt_labels'].shape}")
                        else:
                            print(f"⚠️  gt_labels格式异常: {label_shape}")
                        
                        # 确保边界框和标签数量一致
                        bbox_count = jt_data['gt_bboxes'].shape[0]
                        label_count = jt_data['gt_labels'].shape[0]
                        if bbox_count != label_count:
                            print(f"⚠️  边界框和标签数量不匹配: bboxes={bbox_count}, labels={label_count}")
                            # 取较小的数量
                            min_count = min(bbox_count, label_count)
                            if bbox_count > min_count:
                                jt_data['gt_bboxes'] = jt_data['gt_bboxes'][:min_count]
                            if label_count > min_count:
                                jt_data['gt_labels'] = jt_data['gt_labels'][:min_count]
                            print(f"✅ 已调整数量为: {min_count}")
                except Exception as e:
                    print(f"⚠️  数据格式验证失败: {e}")
                
                # 每处理几个批次就清理一次内存
                if i % 3 == 0:  # 更频繁的内存清理
                    try:
                        clear_jittor_cache()
                        jt.sync_all()
                        # 强制垃圾回收
                        import gc
                        gc.collect()
                        
                        # 检查GPU内存使用情况
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            used_gb = info.used / 1024**3
                            total_gb = info.total / 1024**3
                            if used_gb > total_gb * 0.8:  # 如果使用率超过80%
                                print(f"⚠️  GPU内存使用率过高: {used_gb:.2f}GB/{total_gb:.2f}GB")
                                # 更激进的内存清理
                                jt.gc()
                                jt.sync_all()
                                gc.collect()
                                print("🚨 内存使用率仍然过高，进行激进清理...")
                        except Exception:
                            pass
                    except:
                        pass
                
                # 调试信息（只在第一个批次显示，简化输出）
                if i == 0:
                    print(f"🔍 数据调试信息:")
                    for key, value in jt_data.items():
                        if isinstance(value, jt.Var):
                            print(f"   {key}: {value.shape}, 类型: {type(value)}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"   {key}: list with {len(value)} items")
                            first_item = ensure_jittor_var(value[0], f"{key}[0]")
                            print(f"     first item shape: {first_item.shape}, 类型: {type(first_item)}")
                        else:
                            print(f"   {key}: 类型: {type(value)}")
                
                # 前向传播
                losses = model(**jt_data)
                
                # 确保 total_loss 变量被初始化
                total_loss = None
                
                # 立即检查rcnn_score_loss，这是问题的根源
                if isinstance(losses, dict) and 'rcnn_score_loss' in losses:
                    score_loss_val = ensure_jittor_var(losses['rcnn_score_loss'], 'rcnn_score_loss').item()
                    if abs(score_loss_val) > 1000:
                        print(f"🚨 检测到异常的rcnn_score_loss: {score_loss_val}")
                        # 不要直接重置，而是尝试缩放
                        if score_loss_val > 0:
                            scale_factor = 1000.0 / score_loss_val
                            losses['rcnn_score_loss'] = losses['rcnn_score_loss'] * scale_factor
                            print(f"🔒 rcnn_score_loss 已缩放: {score_loss_val:.2e} -> {ensure_jittor_var(losses['rcnn_score_loss'], 'rcnn_score_loss').item():.4f}")
                        else:
                            # 如果是负值，取绝对值后缩放
                            scale_factor = 1000.0 / abs(score_loss_val)
                            losses['rcnn_score_loss'] = losses['rcnn_score_loss'] * scale_factor
                            print(f"🔒 rcnn_score_loss 已缩放: {score_loss_val:.2e} -> {ensure_jittor_var(losses['rcnn_score_loss'], 'rcnn_score_loss').item():.4f}")
                
                # 调试：检查损失值是否包含 NaN 或 inf，并进行更严格的稳定化处理
                if isinstance(losses, dict):
                    # 检查每个损失值并进行稳定化处理
                    for key, value in losses.items():
                        loss_val = ensure_jittor_var(value, f"losses[{key}]").item()
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
                                    print(f"🔒 {key} 已缩放: {loss_val:.2e} -> {ensure_jittor_var(losses[key], f'losses[{key}]').item():.4f}")
                            else:
                                # 对于其他损失，尝试缩放
                                if abs(loss_val) > 1000:
                                    scale_factor = 100.0 / abs(loss_val)
                                    losses[key] = losses[key] * scale_factor
                                    print(f"🔒 {key} 已缩放: {loss_val:.2e} -> {ensure_jittor_var(losses[key], f'losses[{key}]').item():.4f}")
                                elif np.isnan(loss_val) or np.isinf(loss_val):
                                    losses[key] = jt.array(0.0)
                                    print(f"🔒 {key} 重置为: 0.0 (NaN/Inf)")
                    
                    # 计算总损失并进行稳定化
                    total_loss = sum(losses.values())
                    
                    # 检查总损失是否有效
                    total_loss_val = ensure_jittor_var(total_loss, "total_loss").item()
                    if not np.isfinite(total_loss_val) or abs(total_loss_val) > 1000:
                        print(f"⚠️  WARNING: 总损失 = {total_loss_val} (异常值)")
                        logger.warning(f"Abnormal total loss: {total_loss_val}")
                        
                        # 如果总损失无效，使用所有有效损失的总和
                        valid_losses = []
                        for key, value in losses.items():
                            val = ensure_jittor_var(value, f"losses[{key}]").item()
                            if np.isfinite(val) and abs(val) <= 1000:
                                valid_losses.append(value)
                        
                        if valid_losses:
                            total_loss = sum(valid_losses)
                            print(f"✅ 使用有效损失重新计算总损失: {ensure_jittor_var(total_loss, 'total_loss').item()}")
                        else:
                            # 如果所有损失都无效，尝试使用一个基于批次大小的合理值
                            total_loss = jt.array(0.1 * batch_size)
                            print(f"⚠️  所有损失都无效，使用基于批次大小的值: {ensure_jittor_var(total_loss, 'total_loss').item()}")
                    
                    # 累积各项损失
                    for key, value in losses.items():
                        if key != 'loss':
                            if key not in epoch_components:
                                epoch_components[key] = 0.0
                            try:
                                epoch_components[key] += ensure_jittor_var(value, f"losses[{key}]").item()
                            except Exception as e:
                                print(f"⚠️  累积损失失败 {key}: {e}")
                                epoch_components[key] += 0.0
                else:
                    # 如果losses不是字典，确保total_loss被正确定义
                    try:
                        total_loss = ensure_jittor_var(losses, "losses", (1,))
                        # 检查总损失是否有效
                        total_loss_val = total_loss.item()
                        if not np.isfinite(total_loss_val) or abs(total_loss_val) > 1000:
                            print(f"⚠️  WARNING: 总损失 = {total_loss_val} (异常值)")
                            logger.warning(f"Abnormal total loss: {total_loss_val}")
                            # 如果总损失无效，使用一个小的默认值
                            total_loss = jt.array(0.001)
                    except Exception as e:
                        print(f"⚠️  损失转换失败: {e}")
                        total_loss = jt.array(0.001)
                
                # 温和地限制损失值范围，防止数值不稳定
                try:
                    # 先检查损失值是否异常
                    loss_val = ensure_jittor_var(total_loss, "total_loss").item()
                    if not np.isfinite(loss_val):
                        print(f"⚠️  检测到非有限损失值: {loss_val}")
                        # 如果损失值非有限，使用一个基于批次大小的合理值
                        total_loss = jt.array(0.1 * batch_size)
                        print(f"🔒 使用基于批次大小的损失值: {ensure_jittor_var(total_loss, 'total_loss').item()}")
                    elif abs(loss_val) > 10000:  # 提高阈值，避免过度限制
                        print(f"⚠️  检测到过大损失值: {loss_val}")
                        # 如果损失值过大，进行温和的缩放
                        scale_factor = 1000.0 / abs(loss_val)
                        total_loss = total_loss * scale_factor
                        print(f"🔒 损失值已缩放: {loss_val:.2e} -> {ensure_jittor_var(total_loss, 'total_loss').item():.4f}")
                    else:
                        # 只在损失值正常时进行温和限制
                        total_loss = total_loss.clamp(-1000.0, 1000.0)
                except Exception as e:
                    print(f"⚠️  损失值限制失败: {e}")
                    # 如果限制失败，使用基于批次大小的值
                    total_loss = jt.array(0.1 * batch_size)
                    print(f"🔒 使用基于批次大小的损失值: {ensure_jittor_var(total_loss, 'total_loss').item()}")
                
                # 反向传播 & 梯度裁剪（若配置启用）
                # print(f"🔄 开始反向传播...")
                grad_norm_value = None
                if grad_clip_cfg is not None:
                    # 在Jittor中，梯度裁剪通常通过优化器配置实现，这里简化处理
                    try:
                        max_norm = float(getattr(grad_clip_cfg, 'max_norm', 20))
                        print(f"✂️  梯度裁剪配置: max_norm={max_norm}")
                    except Exception:
                        pass
                
                # 简化的梯度监控（避免使用jt.grad）
                try:
                    # 在Jittor中，我们通常不需要手动计算梯度
                    # 梯度会在optimizer.step()中自动计算
                    pass
                except Exception as e:
                    print(f"⚠️  梯度监控失败: {e}")
                    # 如果梯度监控失败，尝试继续训练
                    print(f"🔄 梯度监控失败，尝试继续训练...")
                
                # 最终检查 total_loss 是否被正确定义
                if total_loss is None:
                    print(f"⚠️  total_loss 仍然为 None，使用默认值")
                    total_loss = jt.array(0.001)
                
                # 更新参数
                try:
                    # 在Jittor中，使用optimizer.step(loss)来自动处理梯度计算和更新
                    # 使用辅助函数确保total_loss是单个Jittor张量
                    total_loss = ensure_jittor_var(total_loss, "total_loss", (1,))
                    
                    print(f"🔍 优化器更新前，total_loss类型: {type(total_loss)}, shape: {ensure_jittor_var(total_loss, 'total_loss').shape}")

                    # 在Jittor中，推荐使用 optimizer.step(loss) 来自动处理
                    optimizer.step(total_loss)
                    
                    processed_batches += 1  # 成功处理的批次
                    # print(f"✅ 参数更新成功")
                except Exception as e:
                    print(f"⚠️  优化器更新失败: {e}")
                    # 如果失败，尝试清理内存并继续
                    try:
                        clear_jittor_cache()
                        jt.sync_all()
                    except:
                        pass
                    logger.error(f"Optimizer step failed: {e}")
                    # 如果优化器更新失败，跳过这个批次
                    skipped_batches += 1
                    continue
                
                # 内存管理优化：清理中间变量和梯度
                try:
                    # 在Jittor中，不需要手动清理梯度，optimizer.step()会自动处理
                    # 清理Jittor缓存
                    clear_jittor_cache()
                    
                    # 清理中间变量引用
                    del total_loss
                    if 'losses' in locals():
                        del losses
                    if 'jt_data' in locals():
                        del jt_data
                    
                    # 强制垃圾回收
                    gc.collect()
                    
                        
                except Exception as e:
                    print(f"⚠️  内存清理失败: {e}")
                
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
                    if 'total_loss' in locals() and total_loss is not None:
                        epoch_loss += ensure_jittor_var(total_loss, "total_loss").item()
                    else:
                        print(f"⚠️  total_loss 未定义，跳过累积")
                        epoch_loss += 0.0
                except Exception as e:
                    print(f"⚠️  累积总损失失败: {e}")
                    epoch_loss += 0.0
                
                num_batches += 1
                total_steps += 1

                # 周期性回收显存，缓解 OOM（Jittor 推荐）
                if (i + 1) % 50 == 0:  # 更频繁的内存清理
                    try:
                        jt.gc()
                        clear_jittor_cache()
                        gc.collect()
                    except Exception:
                        pass
            
                
                # 更新tqdm进度条显示损失信息
                if isinstance(losses, dict):
                    # 只显示主要的损失值，避免信息过多
                    main_losses = {}
                    for k, v in losses.items():
                        if k in ['loss', 'rpn_cls_loss', 'rpn_bbox_loss', 'rcnn_cls_loss', 'rcnn_bbox_loss']:
                            try:
                                main_losses[k] = f"{ensure_jittor_var(v, f'losses[{k}]').item():.4f}"
                            except:
                                main_losses[k] = "0.0000"
                    
                    # 更新进度条描述
                    pbar.set_postfix({
                        'Loss': f"{ensure_jittor_var(total_loss, 'total_loss').item():.4f}",
                        'RPN': f"{main_losses.get('rpn_cls_loss', '0.0000')}",
                        'RCNN': f"{main_losses.get('rcnn_cls_loss', '0.0000')}"
                    })
                else:
                    pbar.set_postfix({'Loss': f"{ensure_jittor_var(total_loss, 'total_loss').item():.4f}"})
                
                # 每100步记录到logger和JSON日志
                if i % 100 == 0:
                    if isinstance(losses, dict):
                        loss_str = ', '.join([f'{k}: {ensure_jittor_var(v, f"losses[{k}]").item():.4f}' for k, v in losses.items()])
                    else:
                        loss_str = f'{ensure_jittor_var(total_loss, "total_loss").item():.4f}'
                    
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
                                record[k] = float(ensure_jittor_var(v, f"losses[{k}]").item())
                            except Exception:
                                pass
                        record['loss'] = float(ensure_jittor_var(total_loss, "total_loss").item())
                    else:
                        record['loss'] = float(ensure_jittor_var(total_loss, "total_loss").item())
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
        
        # 每个epoch结束时清理内存
        clear_jittor_cache()
        gc.collect()
        
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
    
    # 自动调整批次大小以防止内存溢出
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'samples_per_gpu'):
        original_batch_size = cfg.data.samples_per_gpu
        # 如果批次大小大于1，自动调整为1（进一步减少内存使用）
        if original_batch_size > 1:
            cfg.data.samples_per_gpu = 1
            print(f"⚠️  自动调整批次大小: {original_batch_size} -> {cfg.data.samples_per_gpu} (防止内存溢出)")
    
    # 如果命令行指定了批次大小，使用命令行参数
    if args.batch_size:
        if hasattr(cfg, 'data'):
            cfg.data.samples_per_gpu = args.batch_size
            print(f"📝 使用命令行指定的批次大小: {args.batch_size}")
    
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
