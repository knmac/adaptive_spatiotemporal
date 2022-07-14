#!/usr/bin/env bash
# Extract weight from action recognition weight of pipeline 4 <=> top_0 --> global view
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline4_RGB_segs16_ep50_lr0.001_lrst20-40_seed1007__san19pairfreeze112_actreggru2/Oct17_17-38-51/best.model" \
    --output_dir    "pretrained/ucf101/actreggru2_top0_cat__nofc1_1024hid_1lay"
