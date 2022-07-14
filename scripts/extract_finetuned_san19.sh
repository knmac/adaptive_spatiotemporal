#!/usr/bin/env bash
# Extract weight from san19 finetuned with lr 0.003
python tools/extract_san_from_pipeline.py \
    --weight "pretrained/finetuned/san19_epic_112/112_0003.model" \
    --output_dir "pretrained/finetuned/san19_epic_112/"

python tools/extract_san_from_pipeline.py \
    --weight "pretrained/finetuned/san19_epic_224/224_0003.model" \
    --output_dir "pretrained/finetuned/san19_epic_224/"
