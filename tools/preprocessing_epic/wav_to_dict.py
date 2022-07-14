"""Convert from wav file to dictionary

Reference: https://github.com/ekazakos/temporal-binding-network/blob/master/preprocessing_epic/wav_to_dict.py
"""
import sys
import os
import argparse
import pickle
import multiprocessing as mp

import librosa


def read_and_resample(root, file):
    """Read and resample the wav files"""
    samples, sample_rate = librosa.core.load(os.path.join(root, file),
                                             sr=None,
                                             mono=False)

    print(sample_rate)
    return samples, file.split('.')[0]


def parse_args():
    """Parse input argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sound_dir', type=str, help='Directory of EPIC audio')
    parser.add_argument(
        '--output_dir', type=str, help='Directory to save the pickled sound dictionary')
    parser.add_argument(
        '--processes', type=int, default=40, help='Nummber of processes for multiprocessing')

    args = parser.parse_args()
    return args


def main(args):
    """Main function"""
    sound_dict = {}
    process_list = []
    pool = mp.Pool(processes=args.processes)
    for f in os.listdir(args.sound_dir):
        if f.endswith('.wav'):
            p = pool.apply_async(read_and_resample, (args.sound_dir, f))
            process_list.append(p)

    for p in process_list:
        samples, video_name = p.get()
        print(video_name)
        sound_dict[video_name] = samples

    pickle.dump(sound_dict, open(args.output_dir, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
