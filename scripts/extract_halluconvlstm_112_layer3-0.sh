#!/usr/bin/env bash
# Extract weight from hallucination weight with 112x112 resolution at layer3-0
python tools/extract_hallu_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/pipeline3_rgbspec_san19pairfreeze112@layer3-0_halluconvlstm2.sh/best.model" \
    --output_dir    "pretrained/finetuned/halluconvlstm2_112_layer3-0/"


    #--weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/pipeline3_rgbspec_san19pairfreeze112@layer3-0_halluconvlstm2/best.model" \
    #--output_dir    "pretrained/halluconvlstm2_112_layer3-0/"
