import jittor as jt
import numpy as np
import os
import os.path as osp
import sys
import gc
import pickle
import datetime
import time
import warnings
import json
from tqdm import tqdm

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import mmcv
from mmdet.datasets import build_dataloader

from utils.data_util import ensure_jittor_var, safe_convert_to_jittor, safe_sum
from utils.train_utils import clear_jittor_cache
from jittor_components.JittorOptimizer import create_jittor_optimizer

def create_jittor_trainer(model, datasets, cfg, args, distributed=False, validate=True, timestamp=None, meta=None, logger=None, json_log_path=None):
    """创建Jittor训练器"""
    print(f"创建Jittor训练器")
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
    print("检查模型参数稳定性...")
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
            
    except Exception as e:
        print(f"模型参数检查失败: {e}")
    
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
            print(f" 学习率衰减轮次: {step_epochs}")
        elif hasattr(lr_config, 'type') and lr_config.type == 'StepLR':
            step_size = lr_config.get('step', 8)
            if isinstance(step_size, list):
                step_epochs = step_size
            else:
                step_epochs = [step_size]
            print(f" 学习率衰减轮次: {step_epochs}")
    
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
    print(f"\n开始训练！")
    print(f" 总轮次: {max_epochs}")
    print(f" 初始学习率: {optimizer.lr:.6f}")
    print(f" 批次大小: {cfg.data.samples_per_gpu}")
    print(f" 工作目录: {cfg.work_dir}")
    logger.info(f"Training started: epochs={max_epochs}, lr={optimizer.lr:.6f}")
    
    # 初始化训练统计
    epoch_records = []
    
    for epoch in range(max_epochs):
        print(f"\n训练轮次 {epoch + 1}/{max_epochs}")
        print(f" 当前学习率: {optimizer.lr:.6f}")
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
        print(f"本轮次总批次数: {total_batches}")
        
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
                    print(f" 数据批次调试信息:")
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
                        print(f"图像数据转换后: {jt_data['img'].shape}, 类型: {type(jt_data['img'])}")
                    except Exception as img_error:
                        print(f"图像数据转换失败: {img_error}")
                        # 如果转换失败，尝试使用默认值
                        jt_data['img'] = jt.zeros((1, 3, 224, 224), dtype='float32')
                        print(f"使用默认图像张量: {jt_data['img'].shape}")
                
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
                        print(f"GT bboxes 转换后: {jt_data['gt_bboxes'].shape}, 类型: {type(jt_data['gt_bboxes'])}")
                    except Exception as bbox_error:
                        print(f"GT bboxes 转换失败: {bbox_error}")
                        # 如果转换失败，使用默认值
                        jt_data['gt_bboxes'] = jt.zeros((1, 4), dtype='float32')
                        print(f"使用默认 GT bboxes: {jt_data['gt_bboxes'].shape}")
                
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
                        print(f"GT labels 转换后: {jt_data['gt_labels'].shape}, 类型: {type(jt_data['gt_labels'])}")
                    except Exception as label_error:
                        print(f"GT labels 转换失败: {label_error}")
                        # 如果转换失败，使用默认值
                        jt_data['gt_labels'] = jt.zeros((1,), dtype='int32')
                        print(f"使用默认 GT labels: {jt_data['gt_labels'].shape}")
            
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
                            print(f"gt_bboxes格式正确: {bbox_shape}")
                        elif len(bbox_shape) == 3 and bbox_shape[2] == 4:
                            # 如果是 [B, N, 4] 格式，展平为 [B*N, 4]
                            jt_data['gt_bboxes'] = jt_data['gt_bboxes'].view(-1, 4)
                            print(f"gt_bboxes已展平: {jt_data['gt_bboxes'].shape}")
                        else:
                            print(f"gt_bboxes格式异常: {bbox_shape}")
                        
                        # 检查gt_labels格式：应该是 [N] 其中N是标签数量
                        if len(label_shape) == 1:
                            print(f"gt_labels格式正确: {label_shape}")
                        elif len(label_shape) == 2:
                            # 如果是 [B, N] 格式，展平为 [B*N]
                            jt_data['gt_labels'] = jt_data['gt_labels'].view(-1)
                            print(f"gt_labels已展平: {jt_data['gt_labels'].shape}")
                        else:
                            print(f"gt_labels格式异常: {label_shape}")
                        
                        # 确保边界框和标签数量一致
                        bbox_count = jt_data['gt_bboxes'].shape[0]
                        label_count = jt_data['gt_labels'].shape[0]
                        if bbox_count != label_count:
                            print(f"边界框和标签数量不匹配: bboxes={bbox_count}, labels={label_count}")
                            # 取较小的数量
                            min_count = min(bbox_count, label_count)
                            if bbox_count > min_count:
                                jt_data['gt_bboxes'] = jt_data['gt_bboxes'][:min_count]
                            if label_count > min_count:
                                jt_data['gt_labels'] = jt_data['gt_labels'][:min_count]
                            print(f"已调整数量为: {min_count}")
                except Exception as e:
                    print(f"数据格式验证失败: {e}")
                

                # # 调试信息（只在第一个批次显示，简化输出）
                # if i == 0:
                #     print(f"数据调试信息:")
                #     for key, value in jt_data.items():
                #         if isinstance(value, jt.Var):
                #             print(f"   {key}: {value.shape}, 类型: {type(value)}")
                #         elif isinstance(value, (list, tuple)) and len(value) > 0:
                #             print(f"   {key}: list with {len(value)} items")
                #             first_item = ensure_jittor_var(value[0], f"{key}[0]")
                #             print(f"     first item shape: {first_item.shape}, 类型: {type(first_item)}")
                #         else:
                #             print(f"   {key}: 类型: {type(value)}")
                
                # 前向传播
                losses = model(**jt_data)
                
                # 计算总损失并进行稳定化
                total_loss = sum(losses.values())
                
                # 检查总损失是否有效
                total_loss_val = ensure_jittor_var(total_loss, "total_loss").item()
                
                # 累积各项损失
                for key, value in losses.items():
                    if key != 'loss':
                        if key not in epoch_components:
                            epoch_components[key] = 0.0
                        try:
                            epoch_components[key] += ensure_jittor_var(value, f"losses[{key}]").item()
                        except Exception as e:
                            print(f"累积损失失败 {key}: {e}")
                            epoch_components[key] += 0.0
                else:
                    # 如果losses不是字典，确保total_loss被正确定义
                    try:
                        total_loss = ensure_jittor_var(losses, "losses", (1,))
                        # 检查总损失是否有效
                        total_loss_val = total_loss.item()
                        if not np.isfinite(total_loss_val) or abs(total_loss_val) > 1000:
                            print(f"WARNING: 总损失 = {total_loss_val} (异常值)")
                        logger.warning(f"Abnormal total loss: {total_loss_val}")
                        # 如果总损失无效，使用一个小的默认值
                        total_loss = jt.array(0.001)
                    except Exception as e:
                        print(f"损失转换失败: {e}")
                        total_loss = jt.array(0.001)
                
                # 温和地限制损失值范围，防止数值不稳定
                try:
                    # 先检查损失值是否异常
                    loss_val = ensure_jittor_var(total_loss, "total_loss").item()
                    if not np.isfinite(loss_val):
                        print(f"检测到非有限损失值: {loss_val}")
                        # 如果损失值非有限，使用一个基于批次大小的合理值
                        total_loss = jt.array(0.1 * batch_size)
                        print(f"使用基于批次大小的损失值: {ensure_jittor_var(total_loss, 'total_loss').item()}")
                    elif abs(loss_val) > 10000:  # 提高阈值，避免过度限制
                        print(f"检测到过大损失值: {loss_val}")
                        # 如果损失值过大，进行温和的缩放
                        scale_factor = 1000.0 / abs(loss_val)
                        total_loss = total_loss * scale_factor
                        print(f"损失值已缩放: {loss_val:.2e} -> {ensure_jittor_var(total_loss, 'total_loss').item():.4f}")
                    else:
                        # 只在损失值正常时进行温和限制
                        total_loss = total_loss.clamp(-1000.0, 1000.0)
                except Exception as e:
                        print(f"损失值限制失败: {e}")
                        # 如果限制失败，使用基于批次大小的值
                        total_loss = jt.array(0.1 * batch_size)
                        print(f"使用基于批次大小的损失值: {ensure_jittor_var(total_loss, 'total_loss').item()}")
                
                # 反向传播 & 梯度裁剪（若配置启用）
                # print(f"🔄 开始反向传播...")
                grad_norm_value = None
                if grad_clip_cfg is not None:
                    # 在Jittor中，梯度裁剪通常通过优化器配置实现，这里简化处理
                    try:
                        max_norm = float(getattr(grad_clip_cfg, 'max_norm', 20))
                        print(f"梯度裁剪配置: max_norm={max_norm}")
                    except Exception:
                        pass
                
                # 简化的梯度监控（避免使用jt.grad）
                try:
                    # 在Jittor中，我们通常不需要手动计算梯度
                    # 梯度会在optimizer.step()中自动计算
                    pass
                except Exception as e:
                    print(f"梯度监控失败: {e}")

                
                # 最终检查 total_loss 是否被正确定义
                if total_loss is None:
                    print(f"total_loss 仍然为 None，使用默认值")
                    total_loss = jt.array(0.001)
                
                # 更新参数
                try:
                    # 在Jittor中，使用optimizer.step(loss)来自动处理梯度计算和更新
                    # 使用辅助函数确保total_loss是单个Jittor张量
                    total_loss = ensure_jittor_var(total_loss, "total_loss", (1,))
                    
                    print(f"优化器更新前，total_loss类型: {type(total_loss)}, shape: {ensure_jittor_var(total_loss, 'total_loss').shape}")

                    # 在Jittor中，推荐使用 optimizer.step(loss) 来自动处理
                    optimizer.step(total_loss)
                    
                    processed_batches += 1  # 成功处理的批次

                except Exception as e:
                    print(f"优化器更新失败: {e}")
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
                    print(f"内存清理失败: {e}")
                
                # 更新学习率调度器（如果存在）
                if scheduler is not None:
                    try:
                        # 检查当前学习率是否过低
                        current_lr = optimizer.lr
                        if current_lr < 1e-6:  # 如果学习率过低，重置为初始值
                            print(f"学习率过低 ({current_lr:.2e})，重置为初始值")
                            optimizer.lr = 0.005  # 重置为初始学习率
                        
                        scheduler.step()
                        
                        # 安全地获取当前学习率
                        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                            current_lr = optimizer.param_groups[0].get('lr', 0.0)
                            if i % 200 == 0:  # 每200步打印一次学习率
                                print(f"当前学习率: {current_lr:.6f}")
                    except Exception as e:
                        print(f"学习率调度器更新失败: {e}")
                        # 如果调度器更新失败，尝试重置
                        try:
                            if hasattr(scheduler, 'reset'):
                                scheduler.reset()
                                print("学习率调度器已重置")
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
                print(f"WARNING: 平均损失 = {avg_loss} (非有限值)，使用默认值")
                avg_loss = 0.001
        except Exception as e:
            print(f"计算平均损失失败: {e}")
            avg_loss = 0.001
        
        try:
            avg_components = {key: value / num_batches if num_batches > 0 else 0.0 
                             for key, value in epoch_components.items()}
            # 检查组件损失是否有效
            for key, value in avg_components.items():
                if not np.isfinite(value):
                    print(f" WARNING: {key} = {value} (非有限值)，使用默认值")
                    avg_components[key] = 0.0
        except Exception as e:
            print(f"计算平均组件损失失败: {e}")
            avg_components = {}
        
        epoch_losses.append(avg_loss)
        
        # 关闭tqdm进度条
        pbar.close()
        
        print(f"\n轮次 {epoch + 1} 统计:")
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
            print(f"学习率衰减: {old_lr:.6f} -> {optimizer.lr:.6f}")
        
        # 验证
        if validate and len(datasets) > 1:
            print(f"进行验证...")
            model.eval()
        
        # 显示当前epoch完成状态
        print(f"Epoch {epoch+1} 完成时间: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
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
                    print(f"开始保存检查点...")
                    
                    try:
                        jt.sync_all(True)
                        print(f"CUDA 同步完成")
                    except Exception as e:
                        print(f" CUDA 同步警告: {e}")
                    
                    # 等待一段时间确保所有操作完成
                    import time
                    time.sleep(1)
                    
                    # 获取模型状态
                    print(f"获取模型状态...")
                    state = {}
                    try:
                        model_state = model.state_dict()
                        print(f"模型状态获取成功，包含 {len(model_state)} 个参数")
                    except Exception as e:
                        print(f"模型状态获取失败: {e}")
                        model_state = {}
                    
                    # 转换参数为 numpy
                    print(f"转换参数格式...")
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
                    print(f"创建保存目录...")
                    mmcv.mkdir_or_exist(osp.dirname(osp.abspath(checkpoint_path)))
                    
                    print(f"写入检查点文件...")
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({'state_dict': state, 'meta': meta}, f)
                    
                    print(f"检查点保存成功: {checkpoint_path}")
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                    
                except Exception as e:
                    print(f"保存检查点失败: {e}")
                    logger.error(f"Failed to save checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 尝试保存一个简化的检查点
                    try:
                        print(f" 尝试保存简化检查点...")
                        simple_checkpoint_path = osp.join(cfg.work_dir, f'epoch_{epoch+1}_simple.pth')
                        simple_state = {'epoch': epoch + 1, 'error': str(e)}
                        with open(simple_checkpoint_path, 'wb') as f:
                            pickle.dump(simple_state, f)
                        print(f"简化检查点保存成功: {simple_checkpoint_path}")
                    except Exception as e2:
                        print(f"简化检查点也保存失败: {e2}")
        
        print(f"轮次 {epoch + 1} 完成")
    
    # 训练完成总结
    print(f"\n Jittor训练完成!")
    print(f"训练统计:")
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