_base_ = [
    '../_base_/default_runtime.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='CLIPResNet',
        layers=[3, 4, 6, 3],
        style='pytorch'),
    roi_head=dict(
        type='StandardRoIHead',
        shared_head=dict(type='CLIPResLayer', layers=[3, 4, 6, 3]),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='BBoxHeadCLIP',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            with_cls=False,
            reg_class_agnostic=True,
            zeroshot_path='./clip_embeddings/coco_clip_a+cname_rn50_manyprompt.npy',
            num_classes=80,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  ###### the loss_cls here is not appicable in training
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25, ##########
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'),
            max_per_img=100)))


# dataset settings
dataset_type = 'CocoDataset'
data_root = '/root/autodl-tmp/datasets/coco/'
rpl_root = '/root/autodl-tmp/rpl/'
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 400), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'proposals']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['proposals']),
            dict(
                type='ToDataContainer',
                fields=[dict(key='proposals', stack=False)]),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.1@5.0.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline,
        proposal_file= rpl_root + 'cln_coco_rp_train.pkl',
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.1@17.0.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        proposal_file=rpl_root + 'cln_coco_rp_val.pkl'
        ),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.1@5.0.json',
        # img_prefix=data_root + 'train2017/',
        ann_file=data_root + 'annotations/instances_val2017.1@17.0.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        proposal_file=rpl_root + 'cln_coco_rp_val.pkl'
        ))
evaluation = dict(interval=2, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


# learning policy
lr_config = dict(
    policy='step', 
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2, 3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4) 

fp16 = dict(loss_scale=32.)
