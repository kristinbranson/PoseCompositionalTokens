_base_ = ['./fly_bubble_data_20241024.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
find_unused_parameters=False
checkpoint_config = dict(interval=5, create_symlink=False)
evaluation = dict(interval=5, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW', lr=1e-2, betas=(0.9, 0.999), weight_decay=0.15,
                 constructor='SwinLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=[2, 2, 18, 2], layer_decay_rate=0.9,
                                    no_decay_names=['relative_position_bias_table',
                                                    'rpe_mlp',
                                                    'logit_scale']))

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)
total_epochs = 200

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
    ])

data_cfg = dict(
    image_size=[192, 192],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='data/fly_bubble_data_20241024/test_detections.json',
)

# model settings
model = dict(
    type='PCT',
    pretrained='weights/heatmap/swin_base.pth',
    backbone=dict(
        type='SwinV2TransformerRPE2FC',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[16, 16, 16, 8],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        rpe_interpolation='geo',
        use_shift=[True, True, False, False],
        relative_coords_table_type='norm8_log_bylayer',
        attn_type='cosine_mh',
        rpe_output_type='sigmoid',
        postnorm=True,
        mlp_type='normal',
        out_indices=(3,),
        patch_embed_type='normal',
        patch_merge_type='normal',
        strid16=False,
        frozen_stages=5,
    ),
    keypoint_head=dict(
        type='PCT_Head',
        stage_pct='tokenizer',
        in_channels=1024,
        image_size=data_cfg['image_size'],
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(
            type='Classifer_loss',
            token_loss=1.0,
            joint_loss=1.0),
        cls_head=dict(
            conv_num_blocks=2,
            conv_channels=256,
            dilation=1,
            num_blocks=4,
            hidden_dim=64,
            token_inter_dim=64,
            hidden_inter_dim=256,
            dropout=0.0),
        tokenizer=dict(
            guide_ratio=0.0,
            ckpt="",
            encoder=dict(
                drop_rate=0.2,
                num_blocks=4,
                hidden_dim=512,
                token_inter_dim=64,
                hidden_inter_dim=512,
                dropout=0.0,
            ),
            decoder=dict(
                num_blocks=1,
                hidden_dim=32,
                token_inter_dim=64,
                hidden_inter_dim=64,
                dropout=0.0,
            ),
            codebook=dict(
                token_num=34,
                token_dim=512,
                token_class_num=2048,
                ema_decay=0.9,
            ),
            loss_keypoint=dict(
                type='Tokenizer_loss',
                joint_loss_w=1.0, 
                e_loss_w=15.0,
                beta=0.05,)
            )),
    test_cfg=dict(
        flip_test=True,
        dataset_name='fly_bubble_data_20241024'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.1),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img', 'joints_3d', 'joints_3d_visible'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img', 'joints_3d', 'joints_3d_visible'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/fly_bubble_data_20241024'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/train_annotations.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/test_annotations.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/test_annotations.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}})
)
