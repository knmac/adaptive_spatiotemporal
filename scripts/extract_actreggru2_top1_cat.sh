#!/usr/bin/env bash
# Extract weight from action recognition weight with top1_cat
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetuned_pipeline5_112/finetune_pipeline5__top1_nofc1_1024hid_2lay/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_top1_cat__nofc1_1024hid_2lay"
    #--weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/pipeline5_rgbspec_san19pairfreeze_actreggru2_top1_cat/best.model" \
    #--output_dir    "pretrained/actreggru2_top1_cat/"
