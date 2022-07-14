#!/usr/bin/env bash
# Extract actreg `both`-head from actreggru3 to reuse for actreggru2

echo "Top1"
python tools/extract_actregboth_from_pipeline.py \
    --weight        "pretrained/complete/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top1_cat/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_from_actreggru3_top1_cat__nofc1_1024hid_2lay"


echo "Top2"
python tools/extract_actregboth_from_pipeline.py \
    --weight        "pretrained/complete/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top2_cat/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_from_actreggru3_top2_cat__nofc1_1024hid_2lay"


echo "Top3"
python tools/extract_actregboth_from_pipeline.py \
    --weight        "pretrained/complete/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top3_cat/best.model" \
    --output_dir    "pretrained/finetuned/actreggru2_from_actreggru3_top3_cat__nofc1_1024hid_2lay"
