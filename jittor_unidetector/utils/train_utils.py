import jittor as jt
import numpy as np
import os
import gc
import time

class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, optimizer, epoch, filename):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    jt.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """加载检查点"""
    if os.path.exists(filename):
        checkpoint = jt.load(filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {filename}, epoch {epoch}")
        return epoch
    else:
        print(f"Checkpoint {filename} not found")
        return 0

def adjust_learning_rate(optimizer, epoch, lr, schedule):
    """调整学习率"""
    if epoch in schedule:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Learning rate adjusted to {lr}")

def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    maxk = max(topk)
    batch_size = target.shape[0]
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Timer:
    """计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def get_time(self):
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

def create_logger(log_file):
    """创建日志记录器"""
    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def set_random_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

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

def clear_jittor_cache():
    """清理Jittor缓存"""
    try:
        if hasattr(jt, 'core') and hasattr(jt.core, 'clear_cache'):
            jt.core.clear_cache()
        gc.collect()
    except:
        pass

def get_model_info(model):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024  # 假设float32
    }
    
    return info 

def detect_training_stage(cfg, config_path):
    """自动检测训练阶段"""
    
    model_type = cfg.model.type
    if model_type == 'FasterRCNN':
        stage = '1st'
    elif model_type == 'FastRCNN':
        stage = '2nd'
    
    print(f"检测到训练阶段: {stage}")
    return stage