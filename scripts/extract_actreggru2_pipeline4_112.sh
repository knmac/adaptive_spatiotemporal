#!/usr/bin/env bash
# Extract weight from action recognition weight of pipeline 4 <=> top_0
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetuned_pipeline5_112/finetune_pipeline5__top0_nofc1_1024hid_2lay/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_top0_cat__nofc1_1024hid_2lay"
    #--weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/pipeline4_rgbspec_san19pairfreeze112_actreggru2/best.model" \
    #--output_dir    "pretrained/actreggru2_pipeline4_112/"
