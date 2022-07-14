#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/test_pipeline5_rgbspec_san19pairfreeze_actreggru3_top1_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle_full.yaml' \
    --train_cfg         'configs/train_cfgs/test_pipeline8.yaml' \
    --experiment_suffix 'san19pairfreeze_actreggru3_top1' \
    --is_training       false \
    --logdir            'logs' \
    --savedir           'saved_models'
