#!/bin/bash



# Configuration parameters
CONFIG_FILE="configs/singledataset/clip_decouple_faster_rcnn_r50_c4_1x_coco_1ststage.py"
CHECKPOINT_PATH="/root/UniDetector/regionclip_pretrained-cc_rn50_mmdet.pth"
NUM_GPUS=1
WORK_DIR="/root/autodl-tmp/log/clip_decouple_faster_rcnn_r50_c4_1x_coco_1ststage$(date +%Y%m%d_%H%M%S)"
EPOCHS=4
BATCH_SIZE=2
LEARNING_RATE=0.005

# --resume-from /root/autodl-tmp/log/clip_end2end_faster_rcnn_r50_c4_1x_lvis_v0.520250806_133351/latest.pth \
 
# Training command for region proposal stage (CLN model)
tools/dist_train.sh \
    $CONFIG_FILE \
    $NUM_GPUS \
    --work-dir $WORK_DIR \
    --cfg-options \
        load_from=$CHECKPOINT_PATH \
        runner.max_epochs=$EPOCHS \
        data.samples_per_gpu=$BATCH_SIZE \
        optimizer.lr=$LEARNING_RATE \
        log_config.interval=50 \
        lr_config.step="[3,4]"
