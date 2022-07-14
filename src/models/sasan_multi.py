"""Wrapper for multiple modalities with SASAN backbone
"""
import sys
import os
from collections import OrderedDict

import torch
from torch import nn

# from .san import SAN, Bottleneck
from .base_model import BaseModel
from .sasan import SASANResNet26, SASANResNet38, SASANResNet50

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class SASANMulti(BaseModel):

    def __init__(self, device, modality, backbone, stem=False, pretrained_weights=None,
                 new_length=None, new_input_size=None, new_scale_size=None,
                 **kwargs):
        """Initialize model

        Args:
            backbone: `ResNet26`, `ResNet38`, or `ResNet50`
            new_input_size: new input_size (after cropping) if want to resize
            new_scale_size: new scale_size (before cropping) if want to resize
        """
        super(SASANMulti, self).__init__(device)
        self.modality = modality
        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                self.new_length[m] = 1 if (m in ['RGB', 'Spec']) else 5
        else:
            self.new_length = new_length
        self.new_input_size = new_input_size
        self.new_scale_size = new_scale_size

        # Parameters of SAN backbone
        self.stem = stem
        backbone_dict = {
            'ResNet26': SASANResNet26,
            'ResNet38': SASANResNet38,
            'ResNet50': SASANResNet50,
        }
        if backbone in backbone_dict:
            self.backbone_fn = backbone_dict[backbone]
        else:
            raise NotImplementedError

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
                raise 'pretrained_weights must be string or dictionary'
        else:
            self.pretrained_weights = {m: None for m in modality}

        # Prepare SAN basemodels
        self._load_weight_later = []
        self._prepare_base_model()

        # Prepare the spec modalities by replacing the 1st layer
        if 'Spec' in self.modality:
            logger.info('Converting the ImageNet model to a spectrogram init model')
            self.base_model['Spec'] = self._construct_spec_model(self.base_model['Spec'])
            logger.info('Done. Spec model ready.')

        # Load the weights if could not load before
        if len(self._load_weight_later) != 0:
            for m in self._load_weight_later:
                if os.path.isfile(self.pretrained_weights[m]):
                    checkpoint = torch.load(self.pretrained_weights[m])

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
        """Prepare SAN basemodel for each of the modality"""
        self.base_model = OrderedDict()
        self.input_size = OrderedDict()
        self.input_mean = OrderedDict()
        self.input_std = OrderedDict()
        self.feature_dim = 2048  # Feature dimension before final fc layer

        for m in self.modality:
            # Build SAN models
            self.base_model[m] = self.backbone_fn(
                num_classes=1000,  # Final fc will be removed later
                stem=self.stem,
            )
            if self.new_input_size is None:
                self.input_size[m] = 224
            else:
                self.input_size[m] = self.new_input_size
            self.input_std[m] = [1]

            if m == 'RGB':
                self.input_mean[m] = [104, 117, 128]

            # Load pretrained weights
            if self.pretrained_weights[m] is not None:
                if os.path.isfile(self.pretrained_weights[m]):
                    logger.info('Loading pretrained weight for modality: {}'.format(m))
                    checkpoint = torch.load(self.pretrained_weights[m])

                    # Remove `module.` from keys
                    state_dict = {k.replace('module.', ''): v
                                  for k, v in checkpoint['state_dict'].items()}
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

    def _construct_spec_model(self, base_model):
        """Convert ImageNet model to spectrogram init model"""
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Spec'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).contiguous()

        new_conv = nn.Conv2d(self.new_length['Spec'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)

        return base_model

    def forward(self, x, return_concat=True):
        """Forward to get the feature instead of getting the classification

        Args:
            x: dictionary inputs of multiple modalities
            return_concat: whether to concatenate the features

        Return:
            out_feat: concatenated feature output if return_concat is True.
                Otherwise, list of feature output wrt to self.modality
        """
        concatenated = []

        # Get the output for each modality
        for m in self.modality:
            if (m == 'RGB'):
                channel = 3
            elif (m == 'Spec'):
                channel = 1
            sample_len = channel * self.new_length[m]

            base_model = getattr(self, m.lower())
            base_out = base_model(x[m].view((-1, sample_len) + x[m].size()[-2:]))
            concatenated.append(base_out)

        if return_concat:
            out_feat = torch.cat(concatenated, dim=1)
        else:
            out_feat = concatenated
        return out_feat

    def freeze_fn(self, freeze_mode):
        """Copied from tbn model"""
        if freeze_mode == 'modalities':
            for m in self.modality:
                logger.info('Freezing ' + m + ' stream\'s parameters')
                base_model = getattr(self, m.lower())
                for param in base_model.parameters():
                    param.requires_grad_(False)

        elif freeze_mode == 'partialbn_parameters':
            for mod in self.modality:
                count = 0
                logger.info("Freezing BatchNorm2D parameters except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown parameters update in frozen mode
                            m.weight.requires_grad_(False)
                            m.bias.requires_grad_(False)

        elif freeze_mode == 'partialbn_statistics':
            for mod in self.modality:
                count = 0
                logger.info("Freezing BatchNorm2D statistics except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown running statistics update in frozen mode
                            m.eval()
        elif freeze_mode == 'bn_statistics':
            for mod in self.modality:
                logger.info("Freezing BatchNorm2D statistics.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # shutdown running statistics update in frozen mode
                        m.eval()
        else:
            raise ValueError('Unknown mode for freezing the model: {}'.format(freeze_mode))

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        if self.new_scale_size is None:
            scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        else:
            scale_size = {k: self.new_scale_size for k in self.input_size.keys()}
        return scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        if len(self.modality) > 1:
            param_groups = []
            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.rgb.parameters())})
            except AttributeError:
                pass

            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.spec.parameters())})
            except AttributeError:
                pass

            param_groups.append({'params': filter(lambda p: p.requires_grad, self.fusion_classification_net.parameters())})
        else:
            param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
