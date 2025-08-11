import jittor as jt
import jittor.optim as optim
import jittor.lr_scheduler as lr_scheduler
import sys
import os

class JittorOptimizer:
    """Jittor优化器类，用于创建和管理优化器和学习率调度器"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def create_jittor_optimizer(model, cfg):
        """创建Jittor优化器"""
        print("创建Jittor优化器...")
        
        # 从配置文件中读取优化器设置
        optimizer_cfg = cfg.optimizer
        
        # 使用用户配置的学习率，不再强制降低
        base_lr = optimizer_cfg.get('lr', 0.02)
        
        print(f"使用学习率: {base_lr}")
        
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
            print(f"创建SGD优化器，学习率: {base_lr}")
        else:
            print(f"不支持的优化器类型: {optimizer_cfg.type}，使用默认SGD")
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
            print(f"学习率配置: {lr_config}")
            
            # 处理step策略（这是配置文件中使用的策略）
            if hasattr(lr_config, 'policy') and lr_config.policy == 'step':
                step = lr_config.get('step', [3, 4])
                gamma = lr_config.get('gamma', 0.1)
                if isinstance(step, list):
                    milestones = step
                else:
                    milestones = [step]
                
                scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
                print(f"设置StepLR调度器: milestones={milestones}, gamma={gamma}")
                
            elif hasattr(lr_config, 'type') and lr_config.type == 'MultiStepLR':
                milestones = lr_config.get('milestones', [8, 11])
                gamma = lr_config.get('gamma', 0.1)
                scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
                print(f"设置MultiStepLR调度器: milestones={milestones}, gamma={gamma}")
                
            elif hasattr(lr_config, 'type') and lr_config.type == 'StepLR':
                step_size = lr_config.get('step', 8)
                if isinstance(step_size, list):
                    step_size = step_size[0] if len(step_size) > 0 else 8
                gamma = lr_config.get('gamma', 0.1)
                scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                print(f"设置StepLR调度器: step_size={step_size}, gamma={gamma}")
                
            else:
                if hasattr(lr_config, 'type'):
                    print(f"不支持的学习率调度器类型: {lr_config.type}")
                elif hasattr(lr_config, 'policy'):
                    print(f"不支持的学习率策略: {lr_config.policy}")
                else:
                    print("lr_config没有type或policy属性")
                print("学习率调度器未设置，将使用固定学习率")
        else:
            print("配置文件中未找到学习率配置，将使用固定学习率")
        
        # 确保scheduler不为None
        if scheduler is None:
            print("创建默认的学习率调度器（固定学习率）")
            scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)  # 固定学习率
        
        return optimizer, scheduler

# 为了向后兼容，保留原来的函数
def create_jittor_optimizer(model, cfg):
    """创建Jittor优化器（向后兼容函数）"""
    return JittorOptimizer.create_jittor_optimizer(model, cfg)