#!/bin/bash

# Configuration parameters
CONFIG_FILE="configs/singledataset/clip_decouple_faster_rcnn_r50_c4_1x_coco_1ststage.py"
CHECKPOINT_PATH="/root/UniDetector/regionclip_pretrained-cc_rn50_mmdet.pth"
NUM_GPUS=1

tools/dist_test.sh \
    $CONFIG_FILE \
    $NUM_GPUS \
    $CHECKPOINT_PATH\
    --eval bbox
    # --out /root/autodl-tmp/rpl/decouple_oidcoco_lvis_rpn_val.pkl