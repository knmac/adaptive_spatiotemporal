#!/usr/bin/env bash
# Extract weights for nolowres models (top1, 2, 3)

echo "Top1"
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_pipeline5_top1_nofc1_1024hid_2lay_nolowres/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_top1_cat__nofc1_1024hid_2lay_nolowres"

echo "Top2"
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_pipeline5_top2_nofc1_1024hid_2lay_nolowres/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_top2_cat__nofc1_1024hid_2lay_nolowres"

echo "Top3"
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_pipeline5_top3_nofc1_1024hid_2lay_nolowres/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_top3_cat__nofc1_1024hid_2lay_nolowres"
