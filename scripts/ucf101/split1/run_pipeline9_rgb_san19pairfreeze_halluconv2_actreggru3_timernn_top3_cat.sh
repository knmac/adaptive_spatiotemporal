#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/ucf101/pipeline9_rgb_san19pairfreeze_halluconv2_actreggru3_timernn_top3_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/ucf101_split1.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_50_lr0.0003.yaml' \
    --batch_size        24 \
    --seed              1007 \
    --experiment_suffix 'san19pairfreeze_halluconv2_actreggru3_timernn_top3_cat' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'
