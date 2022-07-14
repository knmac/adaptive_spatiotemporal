#!/usr/bin/env bash
# Extract weight from action recognition weight (top1, 2, 3) --> both view

echo "Top1"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze_actreggru2_top1_cat/Oct18_21-13-17/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top1_cat__nofc1_1024hid_1lay"

echo "Top2"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze_actreggru2_top2_cat/Oct25_21-10-11/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top2_cat__nofc1_1024hid_1lay"

echo "Top3"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze_actreggru2_top3_cat/Oct29_11-33-37/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top3_cat__nofc1_1024hid_1lay"
