#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline7_rgbspec_san19pairfreeze_halluconv2_actreggru2_top0_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_100.yaml' \
    --experiment_suffix 'san19pairfreeze_halluconv2_actreggru2_top0_cat' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'


# -----------------------------------------------------------------------------
# Wrapper function
#run_experiment() {
#    _train_mode=$1

#    PYTHONFAULTHANDLER=1 python main.py \
#        --model_cfg         'configs/model_cfgs/pipeline7_rgbspec_san19pairfreeze_halluconv2_actreggru2_top0_cat.yaml' \
#        --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
#        --train_cfg         'configs/train_cfgs/train_san_freeze_adam_100.yaml' \
#        --experiment_suffix 'san19pairfreeze_halluconv2_actreggru2_top0_cat' \
#        --is_training       true \
#        --manual_timestamp  $manual_timestamp \
#        --train_mode        $_train_mode \
#        --logdir            'logs' \
#        --savedir           'saved_models'
#}


# -----------------------------------------------------------------------------
#manual_timestamp=$(date +%b%d_%H-%M-%S)
#max_epoch=100
#finished_epochs=0

#while (( $finished_epochs < $max_epoch )); do
#    if (( $finished_epochs <= 0 )); then
#        # No epoch done, train from scratch
#        run_experiment 'from_scratch'
#    else
#        # Has some epochs, resume from the latest checkpoint
#        run_experiment 'resume'
#    fi

#    # Update the number of finished_epochs
#    finished_epochs=$(ls saved_models/*/$manual_timestamp/epoch_*.model | wc -l)
#    echo '===================================================================='
#done
