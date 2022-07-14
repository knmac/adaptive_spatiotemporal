#!/usr/bin/env bash
# Extract weight from action recognition weight (top1, 2, 3) --> multihead

echo "Top1"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.0001_lrst20-40_seed1007__san19pairfreeze_actreggru3_top1_pretrainedhead/Oct20_22-37-48/best.model" \
    --output_dir    "pretrained/ucf101/actreggru3_top1_cat__nofc1_1024hid_1lay"

echo "Top2"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.0001_lrst20-40_seed1007__san19pairfreeze_actreggru3_top2_pretrainedhead/Oct27_08-39-47/best.model" \
    --output_dir    "pretrained/ucf101/actreggru3_top2_cat__nofc1_1024hid_1lay"

echo "Top3"
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline5_RGB_segs16_ep50_lr0.0001_lrst20-40_seed1007__san19pairfreeze_actreggru3_top3_pretrainedhead/Oct30_21-17-00/best.model" \
    --output_dir    "pretrained/ucf101/actreggru3_top3_cat__nofc1_1024hid_1lay"
