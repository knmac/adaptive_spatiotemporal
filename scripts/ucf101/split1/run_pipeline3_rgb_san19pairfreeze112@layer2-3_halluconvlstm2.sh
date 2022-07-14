#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/ucf101/pipeline3_rgb_san19pairfreeze112@layer2-3_halluconvlstm2.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/ucf101_split1.yaml' \
    --train_cfg         'configs/train_cfgs/ucf101/train_san_freeze_adam_100.yaml' \
    --batch_size        12 \
    --seed              1007 \
    --experiment_suffix 'san19pairfreeze112@layer2-3_halluconvlstm2' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --best_metrics      'val_belief_loss' \
    --best_fn           'min' \
    --logdir            'logs' \
    --savedir           'saved_models'
