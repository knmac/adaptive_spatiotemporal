#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/ucf101/pipeline_simple_san19pair_rgb_112.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/ucf101_split1.yaml' \
    --train_cfg   'configs/train_cfgs/ucf101/train_san_112.yaml' \
    --seed        1007 \
    --batch_size  15 \
    --experiment_suffix 'san19pair_112' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'
