#!/usr/bin/env bash
# Extract weight from SAN19 baselines

# 224x224
echo "extracting 224x224..."
python tools/extract_san_from_pipeline.py \
    --weight        "saved_models/ucf101_PipelineSimple_RGB_segs3_ep75_lr0.00016_lrst_seed1007__san19pair_224/Sep30_17-04-39/best.model" \
    --output_dir    "pretrained/ucf101/san19_224"


# 112x112
echo "extracting 112x112..."
python tools/extract_san_from_pipeline.py \
    --weight        "saved_models/ucf101_PipelineSimple_RGB_segs3_ep100_lr5e-05_lrst40-60_seed1007__san19pair_112/Oct06_12-00-46/best.model" \
    --output_dir    "pretrained/ucf101/san19_112"
