#!/bin/bash

# 混合模式快速启动脚本
# COCO + Objects365 → LVIS 实验

echo "混合模式快速启动 - COCO + Objects365 → LVIS"
echo "=================================================="

python tools/train_stg1.py \
    configs/mixed_mode_coco_objects365_1ststage_rpn.py \
    --work-dir ./work_dirs/mixed_mode_experiment \
    --epochs 1 \
    --seed 1
    