import jittor as jt
import jittor.nn as nn
import sys
import os

# 添加UniDetector的mmdet路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import SHARED_HEADS

@SHARED_HEADS.register_module(force=True)
class CLIPResLayer(nn.Module):
    """
    CLIP ResNet层，用于RoI特征提取
    参考UniDetector的CLIPResLayer实现
    """
    
    def __init__(self, layers=[3, 4, 6, 3], depth=50, **kwargs):
        super(CLIPResLayer, self).__init__()
        
        # 从CLIPResNet中提取layer4
        from ...backbones.clip_backbone import CLIPResNet
        clip_resnet = CLIPResNet(layers=layers)
        
        # 只使用layer4进行RoI特征提取
        self.layer4 = clip_resnet.layer4
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def execute(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            输出特征 [B, C', H', W']
        """
        return self.layer4(x)
    
    def init_weights(self, pretrained=None):
        """初始化权重"""
        pass 