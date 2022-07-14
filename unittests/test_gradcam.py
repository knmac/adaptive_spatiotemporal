#!/usr/bin/env python3
"""Test Grad-CAM"""
import sys
import os
import unittest

import numpy as np
import cv2
import torch
from torchvision import models
from skimage import io
from skimage.transform import resize

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.models.gradcam.grad_cam import GradCam
from src.models.gradcam.grad_cam_bninception import GradCamBNInception
from src.models.gradcam.grad_cam_pipeline import GradCamPipeline
from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory

device = torch.device('cuda')


def get_input(pth='./unittests/both.png', means=[0.485, 0.456, 0.406],
              stds=[0.229, 0.224, 0.225]):
    # Load image
    img = io.imread(pth)
    img = resize(img, (224, 224), anti_aliasing=True, preserve_range=True)
    img = img.astype(np.float32) / 255.0

    # Preprocess
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    return img, preprocessed_img


def show_cam_on_image(img, mask, fname='cam.jpg'):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(fname, np.uint8(255 * cam))


class TestGradCam(unittest.TestCase):
    """Test cases for Grad-CAM"""

    def test_resnet_single_batch(self):
        """Test the modularized version from Gra-CAM implementation"""
        model = models.resnet50(pretrained=True)
        grad_cam = GradCam(device=device, model=model,
                           feature_module=model.layer4, target_layer_names=['2'])
        img, input = get_input()

        model = model.to(device)
        input = input.to(device)
        input.requires_grad_(True)

        mask = grad_cam(input, indices=None, resize=True)
        assert mask.shape == torch.Size([1, 224, 224])

        # mask_ = mask.detach().cpu().numpy()
        # show_cam_on_image(img, mask_[0])

    def test_resnet_multi_batch(self):
        """Test Grad-CAM with multiple samples per batch"""
        model = models.resnet50(pretrained=True)
        grad_cam = GradCam(device=device, model=model,
                           feature_module=model.layer4, target_layer_names=['2'])
        img, input = get_input()
        input = torch.cat([input, input], dim=0)

        model = model.to(device)
        input = input.to(device)
        input.requires_grad_(True)

        mask = grad_cam(input, indices=None, resize=True)
        assert mask.shape == torch.Size([2, 224, 224])
        assert torch.all(mask[0] == mask[1])

        # mask_ = mask.detach().cpu().numpy()
        # show_cam_on_image(img, mask_[0], fname='cam0.jpg')
        # show_cam_on_image(img, mask_[1], fname='cam1.jpg')

    def test_resnet_early_layer(self):
        """Test Grad-CAM with early layer without sub-modules"""
        model = models.resnet50(pretrained=True)
        grad_cam = GradCam(device=device, model=model, feature_module=model.maxpool)
        img, input = get_input()

        model = model.to(device)
        input = input.to(device)
        input.requires_grad_(True)

        mask = grad_cam(input, indices=None, resize=True)
        assert mask.shape == torch.Size([1, 224, 224])

        # mask_ = mask.detach().cpu().numpy()
        # show_cam_on_image(img, mask_[0])

    def test_bninception(self):
        """Test Grad-CAM with BNInception model"""
        full_cfg = {
            'model_name': 'SubBNInception',
            'model_params': {'start_layer': None, 'stop_layer': None},
        }
        model_factory = ModelFactory()

        model = model_factory.generate(
            full_cfg['model_name'], **full_cfg['model_params']).to(device)
        grad_cam = GradCamBNInception(
            device=device, model=model, feature_module='inception_5b_output')
        img, input = get_input()

        model = model.to(device)
        input = input.to(device)
        input.requires_grad_(True)

        mask = grad_cam(input, indices=None, resize=True)
        assert mask.shape == torch.Size([1, 224, 224])

        # mask_ = mask.detach().cpu().numpy()
        # show_cam_on_image(img, mask_[0], fname='cam.jpg')

    def test_pipeline(self):
        """Test GradCam with pipeline"""
        model_cfg = 'configs/model_cfgs/pipeline_simple.yaml'
        model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
        model_factory = ModelFactory()

        # Build pipeline
        model = model_factory.generate(
            model_name, device=device, model_factory=model_factory, **model_params)
        model.to(device)

        # Forward a random input
        b_size = 5
        n_segments = 3
        sample = {
            'RGB': torch.rand([b_size, n_segments*3, 224, 224]).to(device).requires_grad_(True),
            'Flow': torch.rand([b_size, n_segments*10, 224, 224]).to(device).requires_grad_(True),
            'Spec': torch.rand([b_size, n_segments*1, 256, 256]).to(device).requires_grad_(True),
        }

        # GradCAM
        grad_cam = GradCamPipeline(
            device, model, component=model.light_model,
            chosen_modality='RGB', output_type='verb',
            feature_module='inception_4a_output'
        )
        mask, feat = grad_cam(sample, indices=None, resize=True)

        assert mask.shape == torch.Size([b_size, n_segments, 224, 224])
        assert feat.shape == torch.Size([b_size, n_segments, 576, 14, 14])


if __name__ == '__main__':
    unittest.main()
