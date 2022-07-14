#!/usr/bin/env bash
# Extract weights for nolowres models (top1, 2, 3) --> local view

echo "Top1"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze_actreggru2_top1_cat_nolowres/Oct19_08-34-30/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top1_cat__nofc1_1024hid_1lay_nolowres"

echo "Top2"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze_actreggru2_top2_cat_nolowres/Oct26_10-58-55/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top2_cat__nofc1_1024hid_1lay_nolowres"

echo "Top3"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze_actreggru2_top3_cat_nolowres/Oct30_04-49-24/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top3_cat__nofc1_1024hid_1lay_nolowres"
