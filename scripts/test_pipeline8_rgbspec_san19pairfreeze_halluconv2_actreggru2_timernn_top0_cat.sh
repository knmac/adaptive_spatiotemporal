#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/test_pipeline8_rgbspec_san19pairfreeze_halluconv2_actreggru2_timernn_top0_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle_full.yaml' \
    --train_cfg         'configs/train_cfgs/test_pipeline8.yaml' \
    --experiment_suffix 'san19pairfreeze_halluconv2_actreggru2_timernn_top0_cat' \
    --is_training       false \
    --logdir            'logs' \
    --savedir           'saved_models'
