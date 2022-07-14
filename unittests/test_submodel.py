#!/usr/bin/env python3
"""Test sub models"""
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.factories import ModelFactory


class TestSubModel(unittest.TestCase):
    """Test sub models"""

    def __init__(self, *args, **kwargs):
        super(TestSubModel, self).__init__(*args, **kwargs)
        full_cfg = {
            'model_name': 'SubBNInception',
            'model_params': {'start_layer': None, 'stop_layer': None},
        }
        first_half_cfg = {
            'model_name': 'SubBNInception',
            'model_params': {'start_layer': None, 'stop_layer': 'pool2_3x3_s2'},
        }
        second_half_cfg = {
            'model_name': 'SubBNInception',
            'model_params': {'start_layer': 'inception_3a_1x1', 'stop_layer': None},
        }
        device = torch.device('cuda')
        self.device = device

        model_factory = ModelFactory()
        self.full = model_factory.generate(
            full_cfg['model_name'], **full_cfg['model_params']).to(device)
        self.first_half = model_factory.generate(
            first_half_cfg['model_name'], **first_half_cfg['model_params']).to(device)
        self.second_half = model_factory.generate(
            second_half_cfg['model_name'], **second_half_cfg['model_params']).to(device)

    def test_op_list(self):
        full_op_list = self.full._op_list
        first_op_list = self.first_half._op_list
        second_op_list = self.second_half._op_list

        first_op_list.extend(second_op_list)
        assert full_op_list == first_op_list

    def test_channel_dict(self):
        full_channel_dict = self.full._channel_dict
        first_channel_dict = self.first_half._channel_dict
        second_channel_dict = self.second_half._channel_dict

        first_channel_dict.update(second_channel_dict)
        assert full_channel_dict == first_channel_dict

    def test_attribute(self):
        full_attr = set(dir(self.full))
        first_attr = set(dir(self.first_half))
        second_attr = set(dir(self.second_half))

        first_attr.update(second_attr)
        assert full_attr == first_attr

    def test_forward(self):
        sample = torch.rand([1, 3, 224, 224]).to(self.device)
        full_out = self.full(sample)
        first_out = self.first_half(sample)
        second_out = self.second_half(first_out)
        assert torch.all(full_out == second_out)


if __name__ == '__main__':
    unittest.main()
