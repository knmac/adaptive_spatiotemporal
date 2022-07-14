"""Extract hallucination model from a trained pipeline
"""
import sys
import os
import argparse
from collections import OrderedDict

import torch


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weight', type=str,
        help='Trained weight of the pipeline',
    )
    parser.add_argument(
        '-o', '--output_dir', type=str,
        help='Directory to store extracted weights',
    )

    args = parser.parse_args()

    assert os.path.isfile(args.weight), 'Weight not found: {}'.format(args.weight)
    assert not os.path.isdir(args.output_dir), 'output dir already exists'
    os.makedirs(args.output_dir)
    return args


def main(args):
    """Main function"""
    # Load weights
    state_dict = torch.load(args.weight)

    # Extract weights
    actreg_weight = OrderedDict()
    for k in state_dict.keys():
        if k.startswith('actreg_model'):
            new_key = k.replace('actreg_model.', '')
            actreg_weight[new_key] = state_dict[k]

    # Save weights
    torch.save(actreg_weight, os.path.join(args.output_dir, 'actreg.model'))
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
