import jittor as jt
import jittor.nn as nn
import sys
import os

# 将项目根的 mmdet 注册表引入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../..'))
from mmdet.models.builder import NECKS


def _make_conv(in_channels: int, out_channels: int, use_bn: bool = False):
    layers = [nn.Conv2d(in_channels, out_channels, 1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def _make_fpn_conv(channels: int, use_bn: bool = False):
    layers = [nn.Conv2d(channels, channels, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(channels))
    return nn.Sequential(*layers)


@NECKS.register_module(force=True)
class FPN(nn.Module):
    """Jittor 版 FPN，与 MMDet FPN 接口对齐。

    参数基本与 PyTorch 版一致；
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs=False,  # False|'on_input'|'on_lateral'|'on_output'
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg: dict = dict(mode='nearest'),
        init_cfg=None,
    ):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy() if isinstance(upsample_cfg, dict) else dict(mode='nearest')

        # 解析 end_level
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= self.num_ins
            assert num_outs == end_level - start_level

        # 解析 add_extra_convs
        self.add_extra_convs = add_extra_convs
        if isinstance(add_extra_convs, bool) and add_extra_convs:
            self.add_extra_convs = 'on_input'
        if isinstance(self.add_extra_convs, str):
            assert self.add_extra_convs in ('on_input', 'on_lateral', 'on_output')

        # 是否使用 BN
        use_bn = False
        if isinstance(norm_cfg, dict):
            use_bn = norm_cfg.get('type', '').lower() in ('bn', 'batchnorm', 'batchnorm2d')

        # lateral 和 fpn convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(_make_conv(self.in_channels[i], out_channels, use_bn=use_bn and not self.no_norm_on_lateral))
            self.fpn_convs.append(_make_fpn_conv(out_channels, use_bn=use_bn))

        # 额外层（RetinaNet 等）
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_fpn_convs = nn.ModuleList()
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_ch = self.in_channels[self.backbone_end_level - 1] if (i == 0 and self.add_extra_convs == 'on_input') else out_channels
                self.extra_fpn_convs.append(nn.Conv2d(in_ch, out_channels, 3, stride=2, padding=1))

    def _upsample_add(self, x, y):
        # 上采样 x 到 y 大小并相加
        _, _, H, W = y.shape
        x_up = jt.nn.interpolate(x, size=(H, W), mode=self.upsample_cfg.get('mode', 'nearest'))
        return x_up + y

    def execute(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == self.num_ins
        # laterals
        laterals = [self.lateral_convs[i](inputs[i + self.start_level]) for i in range(self.backbone_end_level - self.start_level)]

        # top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self._upsample_add(laterals[i], laterals[i - 1])

        # outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(nn.MaxPool2d(1, stride=2)(outs[-1]))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                else:  # 'on_output'
                    extra_source = outs[-1]
                outs.append(self.extra_fpn_convs[0](extra_source))
                for i in range(1, len(self.extra_fpn_convs)):
                    x = outs[-1]
                    if self.relu_before_extra_convs:
                        x = nn.ReLU()(x)
                    outs.append(self.extra_fpn_convs[i](x))

        return tuple(outs[: self.num_outs])


