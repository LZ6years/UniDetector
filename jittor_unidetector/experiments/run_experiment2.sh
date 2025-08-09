#!/bin/bash

# 实验2：COCO → COCOval 同数据集检测
# 一键执行脚本

set -e  # 遇到错误立即退出

echo "🚀 开始执行实验2：COCO → COCOval"

# 检查环境
echo "📋 检查环境..."
python -c "import jittor; print('✓ Jittor version:', jittor.__version__)"
python -c "import clip; print('✓ CLIP installed successfully')"

# 创建必要的目录
echo "📁 创建目录..."
mkdir -p work_dirs/coco_cln
mkdir -p work_dirs/coco_roi
mkdir -p results
mkdir -p data/proposals

# 数据准备（如果数据不存在，提示用户）
echo "📊 检查数据..."
if [ ! -f "data/coco/annotations/instances_train2017.json" ]; then
    echo "⚠️  请先准备COCO数据集"
    echo "   运行: python tools/prepare_data.py --data-root /path/to/coco --output-dir ./data/coco --dataset coco --split train val"
    exit 1
fi

if [ ! -f "data/coco/annotations/instances_val2017.json" ]; then
    echo "⚠️  请先准备COCO验证集"
    echo "   运行: python tools/prepare_data.py --data-root /path/to/coco --output-dir ./data/coco --dataset coco --split train val"
    exit 1
fi

# 生成CLIP嵌入（如果不存在）
echo "🔤 生成CLIP嵌入..."
if [ ! -f "clip_embeddings/coco_clip_embeddings.npy" ]; then
    echo "生成COCO CLIP嵌入..."
    python tools/generate_clip_embeddings.py \
        --dataset coco \
        --output-path ./clip_embeddings/coco_clip_embeddings.npy
fi

# 第一阶段：训练CLN网络
echo "🎯 第一阶段：训练CLN网络..."
python tools/jittor_train.py \
    --config configs/coco_cln_training.py \
    --work-dir work_dirs/coco_cln \
    --epochs 12 \
    --batch-size 2 \
    --datasets coco

# 生成proposal文件
echo "📋 生成proposal文件..."
python tools/generate_proposals.py \
    --config configs/coco_cln_training.py \
    --checkpoint work_dirs/coco_cln/epoch_12.pkl \
    --output-dir data/proposals

# 第二阶段：训练RoI分类头
echo "🎯 第二阶段：训练RoI分类头..."
python tools/jittor_train.py \
    --config configs/coco_roi_training.py \
    --work-dir work_dirs/coco_roi \
    --epochs 8 \
    --batch-size 2

# 在COCO验证集上测试
echo "🧪 在COCO验证集上测试..."
python tools/test.py \
    --config configs/coco_inference.py \
    --checkpoint work_dirs/coco_roi/epoch_8.pkl \
    --out results/coco_to_coco.json \
    --eval bbox \
    --dataset coco

# 概率校准
echo "⚖️  概率校准..."
python tools/calibrate_results.py \
    --input results/coco_to_coco.json \
    --output results/coco_to_coco_calibrated.json \
    --calibration-method prior_probability

echo "✅ 实验2完成！"
echo "📊 结果文件："
echo "   - 原始结果: results/coco_to_coco.json"
echo "   - 校准结果: results/coco_to_coco_calibrated.json"
echo "   - 模型文件: work_dirs/coco_roi/epoch_8.pkl" 