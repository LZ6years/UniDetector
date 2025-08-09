#!/bin/bash

# 混合模式快速启动脚本
# COCO + Objects365 → LVIS 实验

echo "🚀 混合模式快速启动 - COCO + Objects365 → LVIS"
echo "=================================================="

echo ""
echo "🎯 开始第一阶段训练（RPN）..."
python tools/mixed_mode_train_fixed.py \
    configs/mixed_mode_coco_objects365_1ststage_rpn.py \
    --work-dir ./work_dirs/mixed_mode_experiment \
    --epochs 1 \
    --seed 1
    
echo ""
echo "✅ 第一阶段训练完成！"
echo ""
echo "📁 结果保存在: $WORK_DIR"
echo "📊 下一步可以运行第二阶段训练"
echo ""
echo "🔧 如需继续实验，请运行:"
echo "   python tools/mixed_mode_train.py --config configs/mixed_mode_coco_objects365_2ndstage_roi.py --work-dir $WORK_DIR/stage2_roi" 