#!/usr/bin/env python3
"""Test model"""
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory


class TestModel(unittest.TestCase):
    """Test model loading"""

    def test_tbn_model(self):
        """Test TBN model"""
        model_cfg = 'configs/model_cfgs/tbn.yaml'
        model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
        model_factory = ModelFactory()

        # Build TBN model
        device = torch.device('cuda')
        model = model_factory.generate(model_name, device=device, **model_params)
        model.to(device)

        # Remove dropout to test reproducibility
        model.fusion_classification_net.dropout = 0

        # Forward a random input
        sample = {
            'RGB': torch.rand([1, 9, 224, 224]).to(device),
            'Flow': torch.rand([1, 30, 224, 224]).to(device),
            'Spec': torch.rand([1, 3, 256, 256]).to(device),
        }
        out_run1 = model(sample)
        out_run2 = model(sample)
        assert torch.all(out_run1[0] == out_run2[0])
        assert torch.all(out_run1[1] == out_run2[1])

    def test_pipeline(self):
        """Test simple pipeline"""
        model_cfg = 'configs/model_cfgs/pipeline_simple.yaml'
        model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
        model_factory = ModelFactory()

        # Build pipeline
        device = torch.device('cuda')
        model = model_factory.generate(
            model_name, device=device, model_factory=model_factory, **model_params)
        model.to(device)

        # Remove dropout
        model.actreg_model.dropout = 0

        # Forward a random input
        sample = {
            'RGB': torch.rand([1, 9, 224, 224]).to(device),
            'Flow': torch.rand([1, 30, 224, 224]).to(device),
            'Spec': torch.rand([1, 3, 256, 256]).to(device),
        }
        out_run1 = model(sample)
        out_run2 = model(sample)
        assert torch.all(out_run1[0] == out_run2[0])
        assert torch.all(out_run1[1] == out_run2[1])

    def test_similarity(self):
        """Test to see if the results from simple pipeline is the same as that
        of TBN"""
        model1_cfg = 'configs/model_cfgs/tbn.yaml'
        model1_name, model1_params = ConfigLoader.load_model_cfg(model1_cfg)

        model2_cfg = 'configs/model_cfgs/pipeline_simple.yaml'
        model2_name, model2_params = ConfigLoader.load_model_cfg(model2_cfg)

        model_factory = ModelFactory()

        # Build model
        device = torch.device('cpu')
        model1 = model_factory.generate(
            model1_name, device=device, **model1_params)
        model1.to(device)

        model2 = model_factory.generate(
            model2_name, device=device, model_factory=model_factory, **model2_params)
        model2.to(device)

        # Duplicate some layers because they are initialized randomly
        model2.actreg_model.fc1 = model1.fusion_classification_net.fc1
        model2.actreg_model.fc_verb = model1.fusion_classification_net.fc_verb
        model2.actreg_model.fc_noun = model1.fusion_classification_net.fc_noun
        # model2.actreg_model.fc_action = model1.fusion_classification_net.fc_action

        # Remove dropout to test reproducibility
        model1.fusion_classification_net.dropout = 0
        model2.actreg_model.dropout = 0

        # Forward a random input
        sample = {
            'RGB': torch.rand([1, 9, 224, 224]).to(device),
            'Flow': torch.rand([1, 30, 224, 224]).to(device),
            'Spec': torch.rand([1, 3, 256, 256]).to(device),
        }

        out1 = model1(sample)
        out2 = model2(sample)
        assert torch.all(out1[0] == out2[0])
        assert torch.all(out1[1] == out2[1])

    def test_san_model(self):
        # model_cfg = 'configs/model_cfgs/san10_multi_pairwise.yaml'
        # model_cfg = 'configs/model_cfgs/san10_multi_patchwise.yaml'
        # model_cfg = 'configs/model_cfgs/san15_multi_pairwise.yaml'
        # model_cfg = 'configs/model_cfgs/san15_multi_patchwise.yaml'
        # model_cfg = 'configs/model_cfgs/san19_multi_pairwise.yaml'
        model_cfg = 'configs/model_cfgs/san19_multi_patchwise.yaml'

        model_factory = ModelFactory()
        device = torch.device('cuda')
        model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
        model_params.update({
            'num_segments': 3,
            'modality': ['RGB', 'Flow', 'Spec'],
        })

        model = model_factory.generate(model_name, device=device, **model_params)
        model.to(device)

        sample = {
            'RGB': torch.rand([1, 9, 224, 224]).to(device),
            'Flow': torch.rand([1, 30, 224, 224]).to(device),
            'Spec': torch.rand([1, 3, 256, 256]).to(device),
        }
        out = model(sample)
        assert out.shape == torch.Size([3, 2048*3])

    def test_pipeline_hallu(self):
        """Test pipeline with attention hallucination"""
        model_cfg = 'configs/model_cfgs/pipeline_rgbspec_san10pair_gruhallu.yaml'
        model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
        model_factory = ModelFactory()

        # Build pipeline
        device = torch.device('cuda')
        model = model_factory.generate(
            model_name, device=device, model_factory=model_factory, **model_params)
        model.to(device)

        # Forward a random input
        sample = {
            'RGB': torch.rand([2, 9, 224, 224]).to(device),
            'Spec': torch.rand([2, 3, 256, 256]).to(device),
        }
        model(sample)

        attn = model._attn
        hallu = model._hallu
        assert attn.shape == hallu.shape


if __name__ == '__main__':
    unittest.main()
