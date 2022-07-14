#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline8_rgbspec_san19pairfreeze_halluconv2_actreggru2_timernn_top2_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_50.yaml' \
    --experiment_suffix 'san19pairfreeze_halluconv2_actreggru2_timernn_top2_cat' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'
