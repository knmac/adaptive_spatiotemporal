#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline3_rgbspec_san19pairfreeze112@layer3-0_halluconvlstm2.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_100.yaml' \
    --experiment_suffix 'san19pairfreeze112@layer3-0_halluconvlstm2' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --best_metrics      'val_belief_loss' \
    --best_fn           'min' \
    --logdir            'logs' \
    --savedir           'saved_models'
