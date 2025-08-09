#!/bin/bash

# Jittor UniDetector 快速开始脚本

echo "=== Jittor UniDetector Quick Start ==="

# 1. 环境配置
echo "1. Setting up environment..."
pip install -r requirements.txt

# 2. 数据准备
echo "2. Preparing data..."
python tools/prepare_data.py \
    --data-root /root/autodl-tmp/datasets/coco/ \
    --output-dir ./data \
    --create-small \
    --num-images 1000

# 3. 训练CLN网络（小数据集快速测试）
echo "3. Training CLN network (small dataset for quick test)..."
python tools/train_cln.py \
    --config configs/cln_training.py \
    --work-dir work_dirs/cln_training \
    --epochs 2 \
    --batch-size 2 \
    --lr 0.02

# 4. 测试模型
echo "4. Testing model..."
python tools/test.py \
    --config configs/cln_training.py \
    --checkpoint work_dirs/cln_training/epoch_2.pkl \
    --out test_results.json \
    --eval bbox \
    --show-dir visualizations

echo "=== Quick start completed! ==="
echo "Check the following directories:"
echo "- work_dirs/cln_training/: Training logs and checkpoints"
echo "- test_results.json: Test results"
echo "- visualizations/: Visualization results" 