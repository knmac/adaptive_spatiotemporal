#!/usr/bin/env bash
# Extract weight from hallucination weight with 112x112 resolution at layer3-0
python tools/extract_hallu_from_pipeline.py \
    --weight        "saved_models/ucf101_Pipeline3_RGB_segs16_ep100_lr0.0001_lrst40-80_seed1007__san19pairfreeze112@layer3-0_halluconvlstm2/Oct13_20-16-54/best.model" \
    --output_dir    "pretrained/ucf101/halluconvlstm2_112_layer3-0/"
