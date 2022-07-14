#!/usr/bin/env bash
python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_rgbspec.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg   'configs/train_cfgs/train_tbn.yaml' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'
