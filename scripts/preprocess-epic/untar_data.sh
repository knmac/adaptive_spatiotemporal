#!/usr/bin/env bash
python tools/preprocessing_epic/untar_data.py \
    --input_dir "data/EPIC_KITCHENS_2018/frames_rgb_flow" \
    --output_dir "data/EPIC_KITCHENS_2018/frames_untar"
