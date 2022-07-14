"""Wrapper for multiple modalities with ResNet backbone. Only for feature extraction
"""
import sys
import os
from collections import OrderedDict

import torch
from torchvision import models

from .san_multi import SANMulti

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class ResNetMulti(SANMulti):

    def __init__(self, device, modality, resnet_type, pretrained_weights=None,
                 new_length=None, new_input_size=None, new_scale_size=None,
                 **kwargs):
        super(SANMulti, self).__init__(device)  # Use basemodel init
        self.modality = modality
        self.resnet_type = resnet_type
        self.new_length = OrderedDict()

        if new_length is None:
            for m in self.modality:
                self.new_length[m] = 1 if (m in ['RGB', 'Spec']) else 5
        else:
            self.new_length = new_length
        self.new_input_size = new_input_size
        self.new_scale_size = new_scale_size

        self.san_remove_avgpool = False  # make SANMulti happy

        # Get the pretrained weight and convert to dictionary if neccessary
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if pretrained_weights is not None:
            # If str -> all modalities are init using the same weight
            if isinstance(pretrained_weights, str):
                if not os.path.isfile(pretrained_weights):
                    pretrained_weights = os.path.join(root, pretrained_weights)
                self.pretrained_weights = {m: pretrained_weights for m in modality}
            # If dict -> each modality is init differently
            elif isinstance(pretrained_weights, dict):
                self.pretrained_weights = {}
                for m in modality:
                    if not os.path.isfile(pretrained_weights[m]):
                        pretrained_weights[m] = os.path.join(root, pretrained_weights[m])
                self.pretrained_weights = pretrained_weights
            else:
                raise ValueError('pretrained_weights must be string or dictionary')
        else:
            self.pretrained_weights = {m: None for m in modality}

        # Prepare basemodels
        self._load_weight_later = []
        self._prepare_base_model()

        # Prepare the flow and spec modalities by replacing the 1st layer
        is_flow = any(m == 'Flow' for m in self.modality)
        is_spec = any(m == 'Spec' for m in self.modality)
        if is_flow:
            logger.info('Converting the ImageNet model to a flow init model')
            self.base_model['Flow'] = self._construct_flow_model(self.base_model['Flow'])
            logger.info('Done. Flow model ready...')
        if is_spec:
            logger.info('Converting the ImageNet model to a spectrogram init model')
            self.base_model['Spec'] = self._construct_spec_model(self.base_model['Spec'])
            logger.info('Done. Spec model ready.')

        # Load the weights if could not load before
        if len(self._load_weight_later) != 0:
            for m in self._load_weight_later:
                if os.path.isfile(self.pretrained_weights[m]):
                    checkpoint = torch.load(self.pretrained_weights[m], map_location=self.device)

                    # Remove `module.` from keys
                    state_dict = {k.replace('module.', ''): v
                                  for k, v in checkpoint['state_dict'].items()}
                    self.base_model[m].load_state_dict(state_dict, strict=True)
                    logger.info('Reloaded pretrained weight for SAN modality: {}'.format(m))
                else:
                    logger.info('Not loading pretrained model for modality {}!'.format(m))
        del self._load_weight_later

        # Remove the last fc layer and last avgpool layer
        for m in self.modality:
            last_layer_name = 'fc'
            delattr(self.base_model[m], last_layer_name)

        # Add base models as modules
        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    def _prepare_base_model(self):
        """Prepare ResNet basemodel for each of the modality"""
        self.base_model = OrderedDict()
        self.input_size = OrderedDict()
        self.input_mean = OrderedDict()
        self.input_std = OrderedDict()
        self.feature_dim = 2048  # Feature dimension before final fc layer

        for m in self.modality:
            # Build SAN models
            if self.resnet_type == 'resnet50':
                self.base_model[m] = resnet50_wrapper()
            else:
                raise NotImplementedError

            if self.new_input_size is None:
                self.input_size[m] = 224
            else:
                self.input_size[m] = self.new_input_size
            self.input_std[m] = [1]

            if m == 'Flow':
                self.input_mean[m] = [128]
            elif m == 'RGBDiff':
                self.input_mean[m] = self.input_mean[m] * (1 + self.new_length[m])
            elif m == 'RGB':
                self.input_mean[m] = [104, 117, 128]

            # Load pretrained weights
            if self.pretrained_weights[m] is not None:
                if os.path.isfile(self.pretrained_weights[m]):
                    logger.info('Loading pretrained weight for SAN modality: {}'.format(m))
                    checkpoint = torch.load(self.pretrained_weights[m], map_location=self.device)

                    # Remove `module.` from keys
                    if 'state_dict' in checkpoint:
                        state_dict = {k.replace('module.', ''): v
                                      for k, v in checkpoint['state_dict'].items()}
                    else:
                        state_dict = checkpoint
                    try:
                        self.base_model[m].load_state_dict(state_dict, strict=True)
                    except RuntimeError:
                        logger.info('Cannot load. Will convert and load later...')
                        self._load_weight_later.append(m)
                else:
                    logger.info('Pretrained weights given but not found: {}'.format(
                        self.pretrained_weights[m]))
                    os._exit(-1)
            else:
                logger.info('Not loading pretrained model for modality {}!'.format(m))
                logger.info('Use random init instead')


# =============================================================================
class ResNetWrapper(models.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x


def _resnet_wrapper(block, layers, **kwargs):
    model = ResNetWrapper(block, layers, **kwargs)
    return model


def resnet50_wrapper(**kwargs):
    return _resnet_wrapper(models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
