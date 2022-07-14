#!/usr/bin/env bash
python main.py \
    --model_cfg   'configs/model_cfgs/tbn.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg   'configs/train_cfgs/train_tbn.yaml' \
    --is_training true \
    --train_mode  'from_pretrained' \
    --logdir      'logs' \
    --savedir     'saved_models'
    #--log_fname   'running.log'
