#!/bin/bash

# Simple training script for Object Detection
echo "Starting training..."

# --resume-from /root/autodl-tmp/log/clip_decouple_faster_rcnn_r50_c4_1x_coco_2ndstage20250804_201104/latest.pth \
# clip_decouple_faster_rcnn_r50_c4_1x_objcoco_1ststage

# Training command with common parameters
tools/dist_train.sh \
    configs/inference/clip_end2end_faster_rcnn_r50_c4_1x_lvis_v0.5.py \
    1 \
    --work-dir /root/autodl-tmp/log/clip_end2end_faster_rcnn_r50_c4_1x_lvis_v0.5$(date +%Y%m%d_%H%M%S) \
    --resume-from /root/autodl-tmp/log/clip_end2end_faster_rcnn_r50_c4_1x_lvis_v0.520250806_133351/latest.pth \
    --cfg-options \
        load_from=/root/UniDetector/regionclip_pretrained-cc_rn50_mmdet.pth \
        runner.max_epochs=4 \
        data.samples_per_gpu=2 \
        optimizer.lr=0.005 \
        log_config.interval=50 \
        checkpoint_config.interval=2 \
        seed=1 \
        lr_config.warmup_iters=500 \
        lr_config.warmup_ratio=0.001 \
        lr_config.step="[3,4]"

echo "Training completed!" 