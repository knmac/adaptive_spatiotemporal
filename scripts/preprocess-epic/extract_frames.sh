#!/usr/bin/env bash
# Batch extract frames for all videos in the datasets
# The extracted frames are stored as
#   $OUT_DIR/[video_id]/0/%4d.jpg
# For example
#   EPIC_KITCHENS_2018/frames_256/P01_06/0/0001.jpg
IN_DIR='./data/EPIC_KITCHENS_2018/videos'
OUT_DIR='./data/EPIC_KITCHENS_2018/frames_256'
mkdir -p $OUT_DIR

# Extract function
extract () {
    phase=$1

    for pth1 in $IN_DIR/$phase/*; do
        uid="${pth1##*/}"
        for pth2 in $IN_DIR/$phase/$uid/*; do
            vid="${pth2##*/}"
            output=$OUT_DIR/${vid//\.MP4/}/0

            mkdir -p $output
            ffmpeg \
                -i "$pth2" \
                -vf 'scale=-2:256' \
                -q:v 4 \
                -r 60 \
                "$output/%4d.jpg"
        done
    done
}

echo "========================================================================"
echo "Extracting train videos"
echo "========================================================================"
extract "train"

echo "========================================================================"
echo "Extracting test videos"
echo "========================================================================"
extract "test"
