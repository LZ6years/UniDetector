# Jittor UniDetector - Hybrid Mode Implementation

A Jittor framework implementation of UniDetector using **MMDetection + Jittor hybrid mode**, fully leveraging the advantages of both frameworks to achieve high-performance object detection models.

## Current Implementation Status: Debugging

**Important Notice**: This project is currently in the debugging phase and has encountered several critical issues that need to be resolved.

## Implementation Strategy

### Core Strategy: Hybrid Mode Architecture
This project adopts an innovative **hybrid mode** implementation strategy rather than complete rewriting:

1. **Fully Utilize MMDetection Ecosystem**
   Reuse MMDetection's data loaders, data augmentation, and evaluation tools

2. **Rewrite Jittor Core Components**
   Rewrite CLIP-related backbone networks and classification heads
   Rewrite OLN-related RPN and RoI heads
   Implement Jittor-specific optimizations and memory management

3. **Data Type Conversion and Model Adaptation**
   Perform PyTorch ↔ Jittor tensor conversion at key data flow nodes

## Project Structure

```
jittor_unidetector/
├── models/                          # Core model components
│   ├── backbones/
│   │   └── clip_backbone.py        # CLIP ResNet backbone (Jittor implementation)
│   ├── heads/
│   │   ├── rpn_head.py             # RPN head (Jittor implementation)
│   │   ├── oln_rpn_head.py         # OLN RPN head (Jittor implementation)
│   │   ├── bbox_head.py             # Basic bounding box head (Jittor implementation)
│   │   ├── clip_bbox_head.py        # CLIP bounding box head (Jittor implementation)
│   │   └── roi_heads/
│   │       ├── bbox_heads/
│   │       │   ├── bbox_head_clip_partitioned.py      # Partitioned CLIP head (Jittor)
│   │       │   ├── bbox_head_clip_inference.py        # Inference CLIP head (Jittor)
│   │       │   └── shared2fc_bbox_score_head.py       # Shared FC head (Jittor)
│   │       ├── oln_roi_head.py     # OLN RoI head (Jittor implementation)
│   │       └── shared_heads/
│   │           └── clip_res_layer.py # CLIP shared layer (Jittor implementation)
│   ├── detectors/
│   │   ├── faster_rcnn.py          # FasterRCNN detector (Jittor implementation)
│   │   └── fast_rcnn.py            # FastRCNN detector (Jittor implementation)
│   └── necks/                      # Reuse MMDetection FPN, etc.
├── configs/                         # Configuration files
│   ├── mixed_mode_coco_objects365_1ststage_rpn.py     # First stage RPN configuration
│   ├── mixed_mode_coco_objects365_2ndstage_roi.py     # Second stage RoI configuration
│   └── _base_/
│       └── default_runtime.py      # Basic runtime configuration
├── tools/                          # Training and testing tools
│   └── train_stg1.py              # Hybrid mode training script (core)
├── jittor_components/              # Jittor core components
│   ├── JittorModel.py             # Jittor model creator
│   ├── JittorTrainer.py           # Jittor trainer
│   └── JittorOptimizer.py         # Jittor optimizer
├── utils/                          # Utility functions
├── work_dirs/                      # Working directories
└── docs/                           # Documentation
```

## Current Implementation Challenges
- Data type conversion: Different frameworks have incompatible tensor operation APIs
- Memory management: Mixed usage of Jittor and PyTorch leads to memory accumulation

## Current Runtime Issues Analysis

### Issue 1: Abnormal Loss Values (Loss = 0.0)
**Phenomenon**:
```
2025-08-11 21:35:50,661 - mmdet - WARNING - Abnormal total loss: 0.0
2025-08-11 21:35:51,077 - mmdet - WARNING - Abnormal total loss: 0.0
```

### Issue 2: Type Conversion Problems
**Phenomenon**:
```
Wrong inputs arguments, Please refer to examples(help(jt.ops.array)).

Types of your inputs are:
 self   = module,
 args   = (ndarray, ),

The function declarations are:
 VarHolder* array__(PyObject* obj)
```

### Issue 3: GPU Memory Overflow
**Phenomenon**:
Even with small batch_size and proposal generation regions, GPU memory still explodes