# Jittor UniDetector

基于Jittor框架的UniDetector实现，支持多数据集训练和零样本目标检测。

## 🚨 重要说明：框架兼容性

### 当前实现状态
本项目采用**混合模式**实现：
- **自定义组件**：使用Jittor实现（CLIPResNet, BBoxHeadCLIPPartitioned等）
- **基础组件**：使用MMDetection实现（FPN, RPNHead, StandardRoIHead等）

### 兼容性要求
```bash
# 需要同时安装Jittor和MMDetection
pip install jittor
pip install mmdet  # 用于基础组件
pip install clip
```

### 替代方案
如果你希望完全避免MMDetection依赖，我们提供了以下选择：

1. **简化版本**：只实现核心CLIP组件，使用Jittor + 简化检测器
2. **纯Jittor版本**：需要实现所有基础组件（FPN, RPN等）
3. **混合版本**：当前实现，需要MMDetection支持

## 🚀 框架优势

- **高性能**: Jittor相比PyTorch具有更好的性能和内存效率
- **自动编译**: 支持自动编译优化，提升训练和推理速度
- **混合精度**: 内置自动混合精度支持
- **易用性**: 与PyTorch API兼容，迁移成本低

## 📋 环境要求

### 方案1：混合模式（推荐）
```bash
# 安装Jittor
pip install jittor

# 安装MMDetection（用于基础组件）
pip install mmdet

# 安装CLIP
pip install clip

# 安装其他依赖
pip install numpy opencv-python pycocotools
```

### 方案2：纯Jittor模式（实验性）
```bash
# 只安装Jittor
pip install jittor
pip install clip
pip install numpy opencv-python pycocotools

# 注意：需要实现所有基础组件
```

## 🎯 主要特性

### 1. 多阶段训练架构
- **第一阶段**: 类无关区域提议网络（RPN）
- **第二阶段**: CLIP-based分类头训练

### 2. 自定义组件（Jittor实现）
- `CLIPResNet`: CLIP ResNet骨干网络
- `CLIPResLayer`: CLIP ResNet共享层
- `BBoxHeadCLIPPartitioned`: 分区CLIP分类头
- `BBoxHeadCLIPInference`: 推理专用CLIP头
- `OlnRoIHead`: OLN RoI头

### 3. 基础组件（MMDetection实现）
- `FPN`: 特征金字塔网络
- `RPNHead`: 区域提议网络头
- `StandardRoIHead`: 标准RoI头
- `SingleRoIExtractor`: 单尺度RoI提取器

### 4. 概率校准
- 支持基于类别频率的概率校准
- 提升零样本检测性能

## 🏗️ 项目结构

```
jittor_unidetector/
├── models/
│   ├── backbones/
│   │   └── clip_backbone.py          # CLIP骨干网络（Jittor）
│   ├── necks/                        # 颈部网络（MMDetection）
│   ├── heads/                        # 检测头（MMDetection）
│   ├── roi_heads/
│   │   ├── shared_heads/
│   │   │   └── clip_res_layer.py     # CLIP共享层（Jittor）
│   │   ├── bbox_heads/
│   │   │   ├── bbox_head_clip_partitioned.py  # 分区CLIP头（Jittor）
│   │   │   └── bbox_head_clip_inference.py    # 推理CLIP头（Jittor）
│   │   └── oln_roi_head.py           # OLN RoI头（Jittor）
│   └── detectors/
│       ├── faster_rcnn.py            # FasterRCNN（Jittor）
│       └── fast_rcnn.py              # FastRCNN（Jittor）
├── configs/
│   ├── mixed_mode_coco_objects365_1ststage_rpn.py  # 混合模式第一阶段配置
│   ├── mixed_mode_coco_objects365_2ndstage_roi.py  # 混合模式第二阶段配置
│   ├── coco_objects365_1ststage_rpn.py        # 第一阶段配置（混合）
│   ├── coco_objects365_2ndstage_roi.py        # 第二阶段配置（混合）
│   ├── pure_jittor_coco_objects365_1ststage_rpn.py  # 纯Jittor配置（实验性）
│   └── lvis_inference_with_calibration.py     # LVIS推理配置（混合）
├── tools/
│   ├── mixed_mode_train.py           # 混合模式训练脚本
│   ├── test_mixed_mode.py            # 混合模式测试脚本
│   ├── jittor_train.py               # Jittor训练脚本
│   ├── jittor_test.py                # Jittor测试脚本
│   └── generate_clip_embeddings.py   # CLIP嵌入生成
└── experiments/
    └── run_coco_objects365_lvis_experiment.sh  # 完整实验脚本
```

## 🚀 快速开始（混合模式）

### 1. 环境检查
```bash
# 运行兼容性测试
python tools/test_mixed_mode.py
```

### 2. 一键启动
```bash
# 给脚本执行权限
chmod +x quick_start_mixed_mode.sh

# 运行快速启动脚本
./quick_start_mixed_mode.sh
```

### 3. 分步运行

#### 第一阶段：RPN训练
```bash
python tools/mixed_mode_train.py \
    --config configs/mixed_mode_coco_objects365_1ststage_rpn.py \
    --work-dir ./work_dirs/mixed_mode/stage1_rpn \
    --epochs 4 \
    --batch-size 2
```

#### 第二阶段：RoI训练
```bash
python tools/mixed_mode_train.py \
    --config configs/mixed_mode_coco_objects365_2ndstage_roi.py \
    --work-dir ./work_dirs/mixed_mode/stage2_roi \
    --epochs 4 \
    --batch-size 2
```

## 🎯 实验：COCO + Objects365 → LVIS

### 实验目标
在COCO和Objects365数据集上训练模型，然后在LVISv1数据集上进行零样本检测。

### 实验流程

#### 1. 第一阶段：RPN训练（混合模式）
```bash
# 训练类无关区域提议网络
python tools/mixed_mode_train.py \
    --config configs/mixed_mode_coco_objects365_1ststage_rpn.py \
    --work-dir ./stage1_rpn \
    --epochs 4 \
    --batch-size 2
```

**特点**:
- 使用`OlnRoIHead`进行类无关检测
- 在COCO + Objects365混合数据集上训练
- 生成高质量的region proposals
- **依赖MMDetection的基础组件**

#### 2. 生成Proposals
```bash
# 为所有数据集生成proposals
python tools/generate_proposals.py \
    --config configs/mixed_mode_coco_objects365_1ststage_rpn.py \
    --checkpoint ./stage1_rpn/latest.pkl \
    --output-dir /root/autodl-tmp/rpl \
    --datasets coco objects365 lvis
```

#### 3. 第二阶段：RoI训练（混合模式）
```bash
# 训练CLIP-based分类头
python tools/mixed_mode_train.py \
    --config configs/mixed_mode_coco_objects365_2ndstage_roi.py \
    --work-dir ./stage2_roi \
    --epochs 4 \
    --batch-size 2
```

**特点**:
- 使用`BBoxHeadCLIPPartitioned`进行多数据集训练
- 支持COCO和Objects365的CLIP嵌入
- 使用预生成的proposals进行训练
- **依赖MMDetection的基础组件**

#### 4. LVIS推理（混合模式）
```bash
# 生成原始结果
python tools/jittor_test.py \
    --config configs/lvis_inference_with_calibration.py \
    --checkpoint ./stage2_roi/latest.pkl \
    --out /root/autodl-tmp/rpl/raw_decouple_oidcoco_lvis_rp_val.pkl \
    --eval bbox

# 概率校准推理
python tools/jittor_test.py \
    --config configs/lvis_inference_with_calibration.py \
    --checkpoint ./stage2_roi/latest.pkl \
    --eval bbox
```

**特点**:
- 使用`BBoxHeadCLIPInference`进行推理
- 支持概率校准提升性能
- 零样本检测LVISv1类别
- **依赖MMDetection的基础组件**

### 一键运行
```bash
# 运行完整实验（混合模式）
chmod +x experiments/run_coco_objects365_lvis_experiment.sh
./experiments/run_coco_objects365_lvis_experiment.sh
```

## 🔧 自定义组件详解

### CLIPResNet（Jittor实现）
```python
# 标准CLIP ResNet
backbone=dict(
    type='CLIPResNet',
    layers=[3, 4, 6, 3],
    output_dim=512,
    input_resolution=224,
    width=64,
    pretrained='./clip_weights/RN50.pt')
```

### BBoxHeadCLIPPartitioned（Jittor实现）
```python
# 分区CLIP分类头
bbox_head=dict(
    type='BBoxHeadCLIPPartitioned',
    zeroshot_path=[
        './clip_embeddings/coco_clip_a+cname_rn50_manyprompt.npy', 
        './clip_embeddings/objects365_clip_a+cname_rn50_manyprompt.npy'
    ],
    cat_freq_path=[
        None, 
        '/root/autodl-tmp/datasets/object365/annotations/object365_cat_freq.json'
    ],
    num_classes=365)
```

### BBoxHeadCLIPInference（Jittor实现）
```python
# 推理专用CLIP头
bbox_head=dict(
    type='BBoxHeadCLIPInference',
    zeroshot_path='./clip_embeddings/lvis_clip_a+cname_rn50_manyprompt.npy',
    withcalibration=True,
    resultfile='/root/autodl-tmp/rpl/raw_decouple_oidcoco_lvis_rp_val.pkl',
    gamma=0.3,
    beta=0.8)
```

## 📊 性能优化

### Jittor特定配置
```python
jittor_config = dict(
    use_cuda=True,
    amp_level=3,  # 自动混合精度
    compile_mode='fast'  # 快速编译模式
)
```

### 训练优化
- 使用自动混合精度（AMP）
- 支持梯度累积
- 优化内存使用

## 🎯 实验结果

### 预期性能
- **LVISv1 AP**: ~15-20%
- **LVISv1 AP50**: ~25-30%
- **LVISv1 AP75**: ~10-15%

### 关键改进
1. **概率校准**: 提升稀有类别检测性能
2. **多数据集训练**: 增强模型泛化能力
3. **CLIP嵌入**: 实现零样本检测

## 🔍 故障排除

### 常见问题

#### 1. MMDetection相关错误
```bash
# 错误：ModuleNotFoundError: No module named 'mmdet'
pip install mmdet

# 错误：版本不兼容
pip install mmdet==2.25.0  # 使用兼容版本
```

#### 2. Jittor相关错误
```bash
# 错误：CUDA不可用
python -c "import jittor; print(jittor.cuda.is_available())"

# 错误：版本问题
pip install jittor==1.3.8.15  # 使用稳定版本
```

#### 3. 内存不足
```bash
# 减少batch_size
--batch-size 1

# 使用梯度累积
--accumulate-steps 2
```

### 调试技巧
```bash
# 检查Jittor版本
python -c "import jittor; print(jittor.__version__)"

# 检查MMDetection版本
python -c "import mmdet; print(mmdet.__version__)"

# 检查GPU可用性
python -c "import jittor; print(jittor.cuda.is_available())"

# 检查CLIP安装
python -c "import clip; print('CLIP OK')"

# 运行兼容性测试
python tools/test_mixed_mode.py
```

## 🚀 未来计划

### 纯Jittor版本
- [ ] 实现纯Jittor版本的FPN
- [ ] 实现纯Jittor版本的RPNHead
- [ ] 实现纯Jittor版本的StandardRoIHead
- [ ] 实现纯Jittor版本的数据加载器

### 性能优化
- [ ] 优化Jittor编译策略
- [ ] 实现分布式训练
- [ ] 添加更多Jittor特定优化

## 📚 参考文献

1. [UniDetector: Universal Object Detection](https://arxiv.org/abs/2103.15049)
2. [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
3. [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195)
4. [Jittor: A Just-In-Time Compiler for Deep Learning](https://arxiv.org/abs/2003.09830)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。 