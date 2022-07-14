#!/usr/bin/env bash
# Extract weight from SAN19 trained with 112x112 resolution
python tools/extract_san_from_pipeline.py \
    --weight        "pretrained/san19_epic_otherscales/112/best.model" \
    --output_dir    "pretrained/san19_epic_otherscales/112"
