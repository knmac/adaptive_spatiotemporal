#!/usr/bin/env bash
IN_DIR="./data/UCF101/UCF-101"
OUT_DIR="./data/UCF101/frames_256"
NFRAMES_FILE="./data/UCF101/n_frames_256.txt"
mkdir -p "$OUT_DIR"
rm -f "$NFRAMES_FILE"

for group_pth in $IN_DIR/*; do
    group="${group_pth##*/}"
    group="${group%.*}"

    for pth in $group_pth/*; do
        vid="${pth##*/}"
        vid="${vid%.*}"

        output="$OUT_DIR/$group/$vid"
        mkdir -p $output
        
        # Extract if the output dir is empty -> allow resuming
        if [ -z "$(ls -A $output)" ]; then
            echo $vid
            ffmpeg \
                -i "$pth" \
                -vf 'scale=-2:256' \
                -q:v 4 \
                "$output/%4d.jpg"
        fi

        # Record the number of extracted frames
        nframes=`(ls $output | wc -l)`
        echo "$group/$vid.avi $nframes" >> $NFRAMES_FILE
    done
done
