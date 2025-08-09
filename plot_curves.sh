#!/bin/bash

# Simple script to plot training curves
echo "📊 Starting to plot training curves..."

# Find the latest training log
LOG_DIR="/root/autodl-tmp/log/end2end_faster_rcnn-r50_c4_1x_coco"

LATEST_LOG=$(find $LOG_DIR -name "*.json" -type f | head -1)

# Create output directory
mkdir -p plot_results

# Plot loss curve
echo "📈 Plotting loss curve..."
python tools/analysis_tools/analyze_logs.py plot_curve \
    $LATEST_LOG \
    --keys loss \
    --out plot_results/loss_curve.png \
    --title "Training Loss Curve" \
    --legend "Loss"

# Plot mAP curve
echo "📈 Plotting mAP curve..."
python tools/analysis_tools/analyze_logs.py plot_curve \
    $LATEST_LOG \
    --keys acc \
    --out plot_results/map_curve.png \
    --title "Accuracy Curve" \
    --legend "Accuracy"

# # Plot both curves together
# echo "📈 Plotting combined curves..."
# python tools/analysis_tools/analyze_logs.py plot_curve \
#     $LATEST_LOG \
#     --keys loss bbox_mAP \
#     --out plot_results/training_curves.png \
#     --title "Training Curves"

echo "✅ Plotting completed!"
echo "📁 Results saved in: plot_results/"