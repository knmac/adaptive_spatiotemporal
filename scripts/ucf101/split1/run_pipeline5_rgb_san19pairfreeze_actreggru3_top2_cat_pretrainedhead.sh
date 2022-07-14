#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/ucf101/pipeline5_rgb_san19pairfreeze_actreggru3_top2_cat_pretrainedhead.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/ucf101_split1.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_50_lr0.0001.yaml' \
    --batch_size        28 \
    --seed              1007 \
    --experiment_suffix 'san19pairfreeze_actreggru3_top2_pretrainedhead' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'