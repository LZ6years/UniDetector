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

# 添加父目录到Python路径，以便导入utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

from utils.train_utils import setup_jittor, clear_jittor_cache, detect_training_stage
from utils.data_util import ensure_jittor_var, safe_convert_to_jittor, safe_sum
from jittor_components.JittorModel import create_jittor_compatible_model
from jittor_components.JittorOptimizer import create_jittor_optimizer
from jittor_components.JittorTrainer import create_jittor_trainer


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
    try:
        # 添加父目录到Python路径，以便导入models模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # 导入所有Jittor模型组件，确保它们被注册到mmdet注册表中
        from models.backbones.clip_backbone import CLIPResNet
        from models.heads.roi_heads.oln_roi_head import OlnRoIHead
        from models.heads.roi_heads.bbox_heads.bbox_head_clip_partitioned import BBoxHeadCLIPPartitioned
        from models.heads.roi_heads.bbox_heads.shared2fc_bbox_score_head import Shared2FCBBoxScoreHead
        from models.heads.rpn_head import RPNHead
        from models.heads.oln_rpn_head import OlnRPNHead
        from models.necks.fpn import FPN
        from models.detectors.faster_rcnn import FasterRCNN
        from models.detectors.fast_rcnn import FastRCNN
        from models.heads.roi_heads.roi_extractors.single_roi_extractor import SingleRoIExtractor
        
        print("✅ 成功导入所有Jittor模型组件")
        return True
    except ImportError as e:
        print(f"❌ 导入模型组件失败: {e}")
        return False



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
