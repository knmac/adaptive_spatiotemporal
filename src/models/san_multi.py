"""Wrapper for multiple modalities with SAN backbone. Only for feature extraction
"""
import sys
import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models

from .san import SAN, Bottleneck
from .base_model import BaseModel
from src.utils.misc import MiscUtils

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class SANMulti(BaseModel):

    def __init__(self, device, modality, san_sa_type, san_layers, san_kernels,
                 san_pretrained_weights=None, san_remove_avgpool=False,
                 new_length=None, new_input_size=None, new_scale_size=None,
                 using_cupy=True, **kwargs):
        super(SANMulti, self).__init__(device)
        self.modality = modality
        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                self.new_length[m] = 1 if (m in ['RGB', 'Spec']) else 5
        else:
            self.new_length = new_length
        self.new_input_size = new_input_size
        self.new_scale_size = new_scale_size

        # parameters of SAN backbone
        self.san_sa_type = san_sa_type
        self.san_layers = san_layers
        self.san_kernels = san_kernels
        self.san_remove_avgpool = san_remove_avgpool  # whether to remove the avgpool layer
        self.using_cupy = using_cupy

        # Get the pretrained weight and convert to dictionary if neccessary
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if san_pretrained_weights is not None:
            # If str -> all modalities are init using the same weight
            if isinstance(san_pretrained_weights, str):
                if not os.path.isfile(san_pretrained_weights):
                    san_pretrained_weights = os.path.join(root, san_pretrained_weights)
                self.san_pretrained_weights = {m: san_pretrained_weights for m in modality}
            # If dict -> each modality is init differently
            elif isinstance(san_pretrained_weights, dict):
                self.san_pretrained_weights = {}
                for m in modality:
                    if not os.path.isfile(san_pretrained_weights[m]):
                        san_pretrained_weights[m] = os.path.join(root, san_pretrained_weights[m])
                self.san_pretrained_weights = san_pretrained_weights
            else:
                raise ValueError('san_pretrained_weights must be string or dictionary')
        else:
            self.san_pretrained_weights = {m: None for m in modality}

        # Prepare SAN basemodels
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
                if os.path.isfile(self.san_pretrained_weights[m]):
                    checkpoint = torch.load(self.san_pretrained_weights[m],
                                            map_location=self.device)

                    # Remove `module.` from keys
                    state_dict = {k.replace('module.', ''): v
                                  for k, v in checkpoint['state_dict'].items()}
                    self.base_model[m].load_state_dict(state_dict, strict=True)
                    logger.info('Reloaded pretrained weight for modality: {}'.format(m))
                else:
                    logger.info('Not loading pretrained model for modality {}!'.format(m))
        del self._load_weight_later

        # Remove the last fc layer and last avgpool layer
        for m in self.modality:
            last_layer_name = 'fc'
            delattr(self.base_model[m], last_layer_name)
            if self.san_remove_avgpool:
                delattr(self.base_model[m], 'avgpool')

        # Add base models as modules
        for m in self.base_model.keys():
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
            self.base_model[m] = SAN(
                sa_type=self.san_sa_type,
                block=Bottleneck,
                layers=self.san_layers,
                kernels=self.san_kernels,
                num_classes=1000,  # Final fc will be removed later
                using_cupy=self.using_cupy,
            )
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
            self._load_pretrained_for_modality(m)

    def _load_pretrained_for_modality(self, m):
        """Load pretrained weights for a single modality

        Args:
            m: modality to load
        """
        if self.san_pretrained_weights[m] is not None:
            if os.path.isfile(self.san_pretrained_weights[m]):
                logger.info('Loading pretrained weight for modality: {}'.format(m))
                checkpoint = torch.load(self.san_pretrained_weights[m], map_location=self.device)

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
                raise ValueError('Pretrained weights given but not found: {}'.format(
                    self.san_pretrained_weights[m]))
        else:
            logger.info('Not loading pretrained model for modality {}!'.format(m))
            logger.info('Use random init instead')

    def _construct_flow_model(self, base_model):
        """Covert ImageNet model to flow init model"""
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Flow'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length['Flow'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length['Flow'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

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
        channels_dict = {'RGB': 3, 'Flow': 2, 'Spec': 1}

        # Get the output for each modality
        for m in self.modality:
            channel = channels_dict[m]
            sample_len = channel * self.new_length[m]

            base_model = getattr(self, m.lower())
            base_out = base_model(x[m].view((-1, sample_len) + x[m].size()[-2:]))

            if not self.san_remove_avgpool:
                base_out = base_out.view(base_out.size(0), -1)

            # If avgpool not available, remove the right and bottom row of spec feat
            if self.san_remove_avgpool and m == 'Spec':
                assert base_out.shape[-1] == 8
                base_out = base_out[:, :, :-1, :-1]

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
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.flow.parameters()), 'lr': 0.001})
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


# =============================================================================
class MobileNetWrapper(models.MobileNetV2):
    def __init__(self, **kwargs):
        self.last_channel = 1280
        super(MobileNetWrapper, self).__init__(**kwargs)

        # Create forward hooks to get intermediate outputs of all layers
        self.all_outputs = {}  # store the output of module, wrt to device

        # Hook function to store the module output
        def hook_fn(module, input, output):
            self.all_outputs[input[0].device].append(output)

        # Automatically register forward hooks at all layers
        for _, module in self.features._modules.items():
            module.register_forward_hook(hook_fn)

    def forward(self, x):
        # Clean up the stored output on the correct device
        self.all_outputs[x.device] = []

        # Forward
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        # x = self.classifier(x)  # Remove classifier layer
        return x


def mobilenet_wrapper(**kwargs):
    model = MobileNetWrapper(**kwargs)
    return model
