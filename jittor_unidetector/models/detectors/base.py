import jittor as jt
import jittor.nn as nn

class BaseDetector(nn.Module):
    """检测器基类"""
    
    def __init__(self):
        super(BaseDetector, self).__init__()
    
    def extract_feat(self, img):
        """提取特征"""
        raise NotImplementedError
    
    def forward_train(self, img, img_metas, **kwargs):
        """训练时前向传播"""
        raise NotImplementedError
    
    def simple_test(self, img, img_metas, **kwargs):
        """简单测试"""
        raise NotImplementedError
    
    def aug_test(self, imgs, img_metas, **kwargs):
        """增强测试"""
        raise NotImplementedError
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """测试时前向传播"""
        raise NotImplementedError
    
    def forward(self, img, mode='forward', **kwargs):
        """前向传播"""
        if mode == 'forward':
            return self.forward_test(img, **kwargs)
        elif mode == 'train':
            return self.forward_train(img, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def init_weights(self, pretrained=None):
        """初始化权重"""
        pass
