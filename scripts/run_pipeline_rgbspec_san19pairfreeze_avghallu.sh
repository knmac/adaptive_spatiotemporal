#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline_rgbspec_san19pairfreeze_avghallu.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_50.yaml' \
    --experiment_suffix 'san19pairfreeze_avghallu' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'
