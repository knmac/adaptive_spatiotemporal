#!/usr/bin/env python3
"""Test datasets"""
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils


class TestData(unittest.TestCase):
    """Test dataset loading"""

    def test_epic_kitchens_rgbspec(self):
        """Test epic kitchens dataset with only rgb and spectrogram"""
        dataset_cfg = 'configs/dataset_cfgs/epickitchens.yaml'
        dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
        dataset_factory = DatasetFactory()

        # Prepare some extra parameters
        modality = ['RGB', 'Spec']
        num_segments = 3
        input_mean = {'RGB': [104, 117, 128]}
        input_std = {'RGB': [1], 'Spec': [1]}
        scale_size = {'RGB': 256, 'Spec': 256}
        crop_size = {'RGB': 224, 'Spec': 224}
        new_length = {'RGB': 1, 'Spec': 1}

        # Get augmentation and transforms
        train_augmentation = MiscUtils.get_train_augmentation(modality, crop_size)
        train_transform, val_transform = MiscUtils.get_train_val_transforms(
            modality=modality,
            input_mean=input_mean,
            input_std=input_std,
            scale_size=scale_size,
            crop_size=crop_size,
            train_augmentation=train_augmentation,
        )

        # Create dataset
        dataset = dataset_factory.generate(
            dataset_name, mode='val', modality=modality,
            num_segments=num_segments, new_length=new_length,
            transform=val_transform, **dataset_params,
        )

        assert dataset.name == 'epic_kitchens'

        sample, label = dataset[0]
        # Check shape
        assert label['verb'] == 2
        assert label['noun'] == 10
        assert sample['RGB'].numpy().shape == (9, 224, 224)
        assert sample['Spec'].numpy().shape == (3, 256, 256)

        # Check with data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False,
            num_workers=4, pin_memory=True)

        print('')
        for i, (samples, labels) in enumerate(data_loader):
            print(i, samples['RGB'].shape, samples['Spec'].shape)
            if i >= 2:
                break

    def test_ucf101(self):
        dataset_cfg = 'configs/dataset_cfgs/ucf101_split1.yaml'
        dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
        dataset_factory = DatasetFactory()

        # Prepare some extra parameters
        modality = ['RGB']
        num_segments = 10
        input_mean = {'RGB': [104, 117, 128]}
        input_std = {'RGB': [1]}
        scale_size = {'RGB': 256}
        crop_size = {'RGB': 224}
        new_length = {'RGB': 1}

        # Get augmentation and transforms
        train_augmentation = MiscUtils.get_train_augmentation(modality, crop_size)
        train_transform, val_transform = MiscUtils.get_train_val_transforms(
            modality=modality,
            input_mean=input_mean,
            input_std=input_std,
            scale_size=scale_size,
            crop_size=crop_size,
            train_augmentation=train_augmentation,
        )

        # Create dataset
        dataset = dataset_factory.generate(
            dataset_name, mode='val', modality=modality,
            num_segments=num_segments, new_length=new_length,
            transform=val_transform, **dataset_params,
        )
        sample, label = dataset[0]
        print(sample['RGB'].shape)
        print(label)
        print('')
        # rgb = MiscUtils.deprocess_rgb(sample['RGB'], num_segments)
        # MiscUtils.viz_sequence(rgb, True)


if __name__ == '__main__':
    unittest.main()
