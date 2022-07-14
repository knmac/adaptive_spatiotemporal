"""Restructure the dataset by making symlinks to extracted frames
"""
import sys
import argparse
import glob
from natsort import natsorted
from pathlib import Path


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=Path, help='Directory of epic dataset')
    parser.add_argument(
        '--symlinks_dir', type=Path, help='Directory to save symlinks for EPIC')

    args = parser.parse_args()
    return args


def main(args):
    """Main function"""
    if not args.symlinks_dir.exists():
        args.symlinks_dir.mkdir(parents=True)

    pattern = 'P[0-3][0-9]_[0-9][0-9]/0'
    for source_file in args.data_dir.glob(pattern):
        video = str(source_file).split('/')[-2]
        print(video)

        link_path = args.symlinks_dir / video
        if not link_path.exists():
            link_path.mkdir(parents=True)

        lst = glob.glob(str(source_file / '*'))
        lst = natsorted(lst)

        for i, source in enumerate(lst):
            link = link_path / 'img_{:010d}.jpg'.format(i)
            if link.exists():
                link.unlink()
            link.symlink_to(source)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
