"""Get label meaning wrt sample id (val set) from epickitchens
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils


# Global variables
dataset_cfg = 'configs/dataset_cfgs/epickitchens.yaml'
verb_lbl = 'dataset_splits/EPIC_KITCHENS_2018/EPIC_verb_classes.csv'
noun_lbl = 'dataset_splits/EPIC_KITCHENS_2018/EPIC_noun_classes.csv'

num_segments = 1
modality = ['RGB']
new_length = {'RGB': 1}

model_crop_size = {'RGB': 224}
model_scale_size = {'RGB': 256}
model_input_mean = {'RGB': [104, 117, 128]}
model_input_std = {'RGB': [1]}


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_ids', type=int, nargs='+',
                        default=[3, 17, 57, 87],
                        help='List of sample id to parse')

    args = parser.parse_args()
    return args


def main(args):
    # Load configurations
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
    dataset_params.update({
        'modality': modality,
        'num_segments': num_segments,
        'new_length': new_length,
    })

    # Get training augmentation and transforms
    train_augmentation = MiscUtils.get_train_augmentation(modality, model_crop_size)
    train_transform, val_transform = MiscUtils.get_train_val_transforms(
        modality=modality,
        input_mean=model_input_mean,
        input_std=model_input_std,
        scale_size=model_scale_size,
        crop_size=model_crop_size,
        train_augmentation=train_augmentation,
    )

    # Data loader
    dataset_factory = DatasetFactory()
    val_dataset = dataset_factory.generate(
        dataset_name, mode='val', transform=val_transform, **dataset_params)

    # Get label meaning dictionary
    with open(verb_lbl) as fin:
        content = fin.read().splitlines()[1:]
        verb_dict = {int(line.split(',')[0]): line.split(',')[1] for line in content}

    with open(noun_lbl) as fin:
        content = fin.read().splitlines()[1:]
        noun_dict = {int(line.split(',')[0]): line.split(',')[1] for line in content}

    # parse
    for id in args.sample_ids:
        _, label = val_dataset.__getitem__(id)
        print('id: {}\tlabel: {}\tmeaning: {} {}'.format(
            id, label, verb_dict[label['verb']], noun_dict[label['noun']]))


if __name__ == '__main__':
    sys.exit(main(parse_args()))
