#!/bin/bash

# COCO + Objects365 → LVIS 完整实验脚本
# 基于Jittor框架的UniDetector实现

set -e

echo "🚀 开始COCO + Objects365 → LVIS实验..."

# 检查环境
echo "📋 检查环境..."
python -c "import jittor; print(f'Jittor版本: {jittor.__version__}')"
python -c "import clip; print('CLIP已安装')"

# 创建工作目录
WORK_DIR="/root/autodl-tmp/jittor_unidetector_experiment"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "📁 工作目录: $WORK_DIR"

# 检查数据集
echo "🔍 检查数据集..."
if [ ! -d "/root/autodl-tmp/datasets/coco" ]; then
    echo "❌ COCO数据集未找到，请先下载COCO数据集"
    exit 1
fi

if [ ! -d "/root/autodl-tmp/datasets/object365" ]; then
    echo "❌ Objects365数据集未找到，请先下载Objects365数据集"
    exit 1
fi

if [ ! -d "/root/autodl-tmp/datasets/lvis_v1.0" ]; then
    echo "❌ LVISv1数据集未找到，请先下载LVISv1数据集"
    exit 1
fi

# 检查CLIP权重
echo "🔍 检查CLIP权重..."
if [ ! -f "./clip_weights/RN50.pt" ]; then
    echo "📥 下载CLIP权重..."
    mkdir -p ./clip_weights
    wget -O ./clip_weights/RN50.pt https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca6b2d6c9c3f3b444f8b0b0b0b0b0b0b/RN50.pt
fi

# 检查CLIP嵌入
echo "🔍 检查CLIP嵌入..."
if [ ! -f "./clip_embeddings/coco_clip_a+cname_rn50_manyprompt.npy" ]; then
    echo "📥 生成COCO CLIP嵌入..."
    mkdir -p ./clip_embeddings
    python ../tools/generate_clip_embeddings.py \
        --dataset coco \
        --output ./clip_embeddings/coco_clip_a+cname_rn50_manyprompt.npy
fi

if [ ! -f "./clip_embeddings/objects365_clip_a+cname_rn50_manyprompt.npy" ]; then
    echo "📥 生成Objects365 CLIP嵌入..."
    python ../tools/generate_clip_embeddings.py \
        --dataset objects365 \
        --output ./clip_embeddings/objects365_clip_a+cname_rn50_manyprompt.npy
fi

if [ ! -f "./clip_embeddings/lvis_clip_a+cname_rn50_manyprompt.npy" ]; then
    echo "📥 生成LVIS CLIP嵌入..."
    python ../tools/generate_clip_embeddings.py \
        --dataset lvis \
        --output ./clip_embeddings/lvis_clip_a+cname_rn50_manyprompt.npy
fi

# 第一阶段：RPN训练
echo "🎯 第一阶段：RPN训练（类无关区域提议）..."
python ../tools/jittor_train.py \
    --config ../configs/coco_objects365_1ststage_rpn.py \
    --work-dir ./stage1_rpn \
    --epochs 4 \
    --batch-size 2

echo "✅ 第一阶段训练完成"

# 生成proposals
echo "📊 生成proposals..."
python ../tools/generate_proposals.py \
    --config ../configs/coco_objects365_1ststage_rpn.py \
    --checkpoint ./stage1_rpn/latest.pkl \
    --output-dir /root/autodl-tmp/rpl \
    --datasets coco objects365 lvis

echo "✅ Proposals生成完成"

# 第二阶段：RoI训练
echo "🎯 第二阶段：RoI训练（CLIP分类头）..."
python ../tools/jittor_train.py \
    --config ../configs/coco_objects365_2ndstage_roi.py \
    --work-dir ./stage2_roi \
    --epochs 4 \
    --batch-size 2

echo "✅ 第二阶段训练完成"

# 生成原始结果（不带概率校准）
echo "📊 生成原始结果（不带概率校准）..."
python ../tools/jittor_test.py \
    --config ../configs/lvis_inference_with_calibration.py \
    --checkpoint ./stage2_roi/latest.pkl \
    --out /root/autodl-tmp/rpl/raw_decouple_oidcoco_lvis_rp_val.pkl \
    --eval bbox

echo "✅ 原始结果生成完成"

# 概率校准推理
echo "📊 概率校准推理..."
python ../tools/jittor_test.py \
    --config ../configs/lvis_inference_with_calibration.py \
    --checkpoint ./stage2_roi/latest.pkl \
    --eval bbox

echo "✅ 概率校准推理完成"

# 结果分析
echo "📈 结果分析..."
python ../tools/analyze_results.py \
    --result-file /root/autodl-tmp/rpl/raw_decouple_oidcoco_lvis_rp_val.pkl \
    --gt-file /root/autodl-tmp/datasets/lvis_v1.0/annotations/lvis_v1_val.1@4.0.json

echo "🎉 实验完成！"
echo "📁 结果保存在: $WORK_DIR"
echo "📊 详细结果请查看: /root/autodl-tmp/rpl/" 