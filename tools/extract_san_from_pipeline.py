"""Extract SAN models from a trained pipeline and format the keys to be loaded
again as pretrained weight"""
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


def add_dummy_final_fc(state_dict, fc_in=2048, fc_out=1000):
    """Add dummy weights for fc layer because the pipeline has removed this"""
    state_dict['state_dict']['module.fc.weight'] = torch.zeros([fc_out, fc_in])
    state_dict['state_dict']['module.fc.bias'] = torch.zeros([fc_out])


def main(args):
    """Main function"""
    # Load weights
    state_dict = torch.load(args.weight)

    # Extract weights
    rgb_weight = {'state_dict': OrderedDict()}
    flow_weight = {'state_dict': OrderedDict()}
    spec_weight = {'state_dict': OrderedDict()}

    for k in state_dict.keys():
        if k.startswith('light_model.rgb.'):
            new_key = k.replace('light_model.rgb.', 'module.')
            rgb_weight['state_dict'][new_key] = state_dict[k]
        elif k.startswith('light_model.flow.'):
            new_key = k.replace('light_model.flow.', 'module.')
            flow_weight['state_dict'][new_key] = state_dict[k]
        elif k.startswith('light_model.spec.'):
            new_key = k.replace('light_model.spec.', 'module.')
            spec_weight['state_dict'][new_key] = state_dict[k]

    # Save weights
    if len(rgb_weight['state_dict']) != 0:
        add_dummy_final_fc(rgb_weight)
        torch.save(rgb_weight, os.path.join(args.output_dir, 'rgb.model'))
    if len(flow_weight['state_dict']) != 0:
        add_dummy_final_fc(flow_weight)
        torch.save(flow_weight, os.path.join(args.output_dir, 'flow.model'))
    if len(spec_weight['state_dict']) != 0:
        add_dummy_final_fc(spec_weight)
        torch.save(spec_weight, os.path.join(args.output_dir, 'spec.model'))
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
