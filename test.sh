#!/bin/bash

# Simple testing script for Object Detection
echo "Starting testing..."

# clip_decouple_faster_rcnn_r50_c4_1x_objcoco_2ndstage20250805_144443
# clip_decouple_faster_rcnn_r50_c4_1x_objcoco_2ndstage_rpn20250805_232222

# Testing command with common parameters
tools/dist_test.sh \
    configs/inference/clip_decouple_faster_rcnn_r50_c4_1x_lvis_v0.5_2ndstage.py \
    /root/autodl-tmp/log/clip_decouple_faster_rcnn_r50_c4_1x_objcoco_2ndstage_rpn20250805_232222/latest.pth \
    1 \
    --eval bbox
# *    --out /root/autodl-tmp/rpl/decouple_oidcoco_lvis_rpn_val.pkl \*

echo "Testing completed!" 