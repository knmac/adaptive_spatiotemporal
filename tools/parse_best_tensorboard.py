"""Parse the tensorboard quickly to find the best prec@1
"""
import sys
import os
import glob
import argparse

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--log_dir', type=str)

    args = parser.parse_args()
    assert os.path.isdir(args.log_dir)
    return args


def get_scalars(log_dir, log_type, verbose=False):
    event_fnames = glob.glob(os.path.join(log_dir, log_type, 'event*'))
    event_fnames.sort()
    if len(event_fnames) == 0:
        if verbose:
            print('  {} --> len(event_fname)=={}'.format(log_type, len(event_fnames)))
        return None

    scalars = None
    for i in range(len(event_fnames)):
        ttf_guidance = {
            event_accumulator.SCALARS: 0,
        }
        event_fname = event_fnames[i]
        ea = event_accumulator.EventAccumulator(event_fname, ttf_guidance)
        ea.Reload()

        if scalars is None:
            scalars = [ea.Scalars(tag) for tag in ea.Tags()['scalars']]
        else:
            for j, tag in enumerate(ea.Tags()['scalars']):
                scalars[j] += ea.Scalars(tag)

    return scalars


def main(args):
    """Main function"""
    # Find argmax wrt data_prec_top1
    log_type = 'data_prec_top1_validation'
    top1 = get_scalars(args.log_dir, log_type)[0]
    argmax = np.argmax([item.value for item in top1])
    print('n_epochs =', len(top1), ', argmax =', argmax)

    # Print all data at argmax
    log_types = [
        'data_loss_validation',
        'data_prec_top1_validation',
        'data_prec_top5_validation',
        'data_verb_loss_validation',
        'data_verb_prec_top1_validation',
        'data_verb_prec_top5_validation',
        'data_noun_loss_validation',
        'data_noun_prec_top1_validation',
        'data_noun_prec_top5_validation',
        'data_gflops_total_validation',
        'data_gflops_avg_validation',
        'data_n_frames_skipped_validation',
        'data_n_frames_prescanned_validation',
        'data_n_frames_nonskipped_validation',
    ]
    for log_type in log_types:
        data = get_scalars(args.log_dir, log_type)
        if data is None:
            continue
        data = data[0]
        print('{} --> {:.03f}'.format(log_type, data[argmax].value))
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
