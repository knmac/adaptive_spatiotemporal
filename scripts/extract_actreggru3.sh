#!/usr/bin/env bash
# Extract weight from action recognition weight with multihead (actreggru3)

#echo "Top1"
#python tools/extract_actreg_from_pipeline.py \
#    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top1_cat/best.model" \
#    --output_dir    "pretrained/finetuned/actreggru3_top1_cat__nofc1_1024hid_2lay"

#echo "Top2"
#python tools/extract_actreg_from_pipeline.py \
#    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top2_cat/best.model" \
#    --output_dir    "pretrained/finetuned/actreggru3_top2_cat__nofc1_1024hid_2lay"

#echo "Top3"
#python tools/extract_actreg_from_pipeline.py \
#    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top3_cat/best.model" \
#    --output_dir    "pretrained/finetuned/actreggru3_top3_cat__nofc1_1024hid_2lay"

echo "Top1"
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_spatial_nms_run_pipeline5_top1_64_reorder/best.model" \
    --output_dir    "pretrained/finetuned/actreggru3_top1_cat_nms__nofc1_1024hid_2lay"

echo "Top2"
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_spatial_nms_run_pipeline5_top2_64_reorder/best.model" \
    --output_dir    "pretrained/finetuned/actreggru3_top2_cat_nms__nofc1_1024hid_2lay"

echo "Top3"
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_spatial_nms_run_pipeline5_top3_64_reorder/best.model" \
    --output_dir    "pretrained/finetuned/actreggru3_top3_cat_nms__nofc1_1024hid_2lay"
