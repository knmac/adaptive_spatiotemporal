#!/usr/bin/env bash
python tools/extract_san_from_pipeline.py \
    --weight 'saved_models/epic_kitchens_PipelineSimple_RGBSpec_segs3_dr0.5_ep100_lr0.01_lr_st30_60_90_/san19pair/best.model' \
    --output_dir 'pretrained/san19_epic_07-10'
