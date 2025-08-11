# Jittor UniDetector - 混合模式复现

基于Jittor框架的UniDetector实现，采用**MMDetection + Jittor混合模式**，充分利用两个框架的优势，实现高性能的目标检测模型。

## 当前复现状态：调试中

**重要提醒**：本项目目前处于调试阶段，遇到了多个关键问题需要解决。

## 复现思路

### 核心策略：混合模式架构
本项目采用创新的**混合模式**实现策略，而非完全重写：

1. **充分利用MMDetection生态**
   复用MMDetection的数据加载器、数据增强、评估工具

2. **Jittor核心组件重写**
   重写CLIP相关的骨干网络和分类头
   重写OLN相关的RPN和RoI头
   实现Jittor特定的优化和内存管理

3. **数据类型转换和模型适配**
   在数据流关键节点进行PyTorch ↔ Jittor张量转换

## 项目结构

```
jittor_unidetector/
├── models/                          # 核心模型组件
│   ├── backbones/
│   │   └── clip_backbone.py        # CLIP ResNet骨干网络（Jittor实现）
│   ├── heads/
│   │   ├── rpn_head.py             # RPN头（Jittor实现）
│   │   ├── oln_rpn_head.py         # OLN RPN头（Jittor实现）
│   │   ├── bbox_head.py             # 基础边界框头（Jittor实现）
│   │   ├── clip_bbox_head.py        # CLIP边界框头（Jittor实现）
│   │   └── roi_heads/
│   │       ├── bbox_heads/
│   │       │   ├── bbox_head_clip_partitioned.py      # 分区CLIP头（Jittor）
│   │       │   ├── bbox_head_clip_inference.py        # 推理CLIP头（Jittor）
│   │       │   └── shared2fc_bbox_score_head.py       # 共享FC头（Jittor）
│   │       ├── oln_roi_head.py     # OLN RoI头（Jittor实现）
│   │       └── shared_heads/
│   │           └── clip_res_layer.py # CLIP共享层（Jittor实现）
│   ├── detectors/
│   │   ├── faster_rcnn.py          # FasterRCNN检测器（Jittor实现）
│   │   └── fast_rcnn.py            # FastRCNN检测器（Jittor实现）
│   └── necks/                      # 复用MMDetection的FPN等
├── configs/                         # 配置文件
│   ├── mixed_mode_coco_objects365_1ststage_rpn.py     # 第一阶段RPN配置
│   ├── mixed_mode_coco_objects365_2ndstage_roi.py     # 第二阶段RoI配置
│   └── _base_/
│       └── default_runtime.py      # 基础运行时配置
├── tools/                          # 训练和测试工具
│   └── train_stg1.py              # 混合模式训练脚本（核心）
├── jittor_components/              # Jittor核心组件
│   ├── JittorModel.py             # Jittor模型创建器
│   ├── JittorTrainer.py           # Jittor训练器
│   └── JittorOptimizer.py         # Jittor优化器
├── utils/                          # 工具函数
├── work_dirs/                      # 工作目录
└── docs/                           # 文档
```

## 当前复现遇到的挑战
- 数据类型转换: 不同框架的张量操作API不完全兼容
- 内存管理: Jittor和PyTorch混合使用导致内存累积


## 当前运行问题分析

### 问题1：损失值异常（Loss = 0.0）
**现象**：
```
2025-08-11 21:35:50,661 - mmdet - WARNING - Abnormal total loss: 0.0
2025-08-11 21:35:51,077 - mmdet - WARNING - Abnormal total loss: 0.0
```

### 问题2：类型转化问题
**现象**：
```
 Wrong inputs arguments, Please refer to examples(help(jt.ops.array)).

Types of your inputs are:
 self   = module,
 args   = (ndarray, ),

The function declarations are:
 VarHolder* array__(PyObject* obj)
```

### 问题3: GPU显存溢出
**现象**：
即使batch_size和生成proprosal的区域已经足够小了,但是还是不行显存爆炸