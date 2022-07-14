"""Extract audio from video files

Reference: https://github.com/ekazakos/temporal-binding-network/blob/master/preprocessing_epic/extract_audio.py
"""
import sys
import os
import argparse
import subprocess


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--videos_dir', type=str, help='Directory of EPIC videos with audio')
    parser.add_argument(
        '--output_dir', type=str, help='Directory of EPIC videos with audio')
    parser.add_argument(
        '--sample_rate', type=str, default='24000', help='Rate to resample audio')

    args = parser.parse_args()
    return args


def ffmpeg_extraction(input_video, output_sound, sample_rate):
    """Extract audio from video

    Args:
        input_video: path to the input video
        output_sound: path to the output sound
        sample_rate: sampling rate to extract the audio
    """
    ffmpeg_command = ['ffmpeg', '-i', input_video,
                      '-vn', '-acodec', 'pcm_s16le',
                      '-ac', '1', '-ar', sample_rate,
                      output_sound]

    subprocess.call(ffmpeg_command)


def main(args):
    """Main function"""
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for root, dirs, files in os.walk(args.videos_dir):
        for f in files:
            if f.endswith('.MP4'):
                input_video = os.path.join(root, f)
                output_sound = os.path.join(
                    args.output_dir, os.path.splitext(f)[0]+'.wav')
                if os.path.isfile(output_sound):
                    print('{} exists'.format(output_sound))
                else:
                    ffmpeg_extraction(input_video, output_sound, args.sample_rate)
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
