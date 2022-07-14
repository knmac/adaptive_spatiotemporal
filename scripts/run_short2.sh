#!/usr/bin/env bash
echo "Testing a short run"

PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_san19pair_rgbspec.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens_short.yaml' \
    --train_cfg   'configs/train_cfgs/train_tbn_short2.yaml' \
    --is_training true \
    --train_mode  'from_pretrained' \
    --logdir      'logs' \
    --savedir     'saved_models'
