import sys
import os

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from .base_model import BaseModel
from .pytorch_ssim.ssim import SSIM
from src.utils.load_cfg import ConfigLoader


class Pipeline(BaseModel):
    """Simple pipeline with spatial and temporal sampler"""
    def __init__(self, device, model_factory, num_class, num_segments, modality,
                 new_length, dropout, attention_layer, attention_dim,
                 light_model_cfg, heavy_model_cfg,
                 time_sampler_cfg, space_sampler_cfg, actreg_model_cfg,
                 hallu_loss='mse'):
        super(Pipeline, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.dropout = dropout
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.hallu_loss = hallu_loss

        # Generate models
        name, params = ConfigLoader.load_model_cfg(light_model_cfg)
        params.update({
            'new_length': self.new_length,
            'dropout': self.dropout,
            'num_class': self.num_class,
            'num_segments': self.num_segments,
            'modality': self.modality,
        })
        self.light_model = model_factory.generate(name, device=device, **params)

        name, params = ConfigLoader.load_model_cfg(heavy_model_cfg)
        params['modality'] = self.modality
        self.heavy_model = model_factory.generate(name, device=device, **params)

        name, params = ConfigLoader.load_model_cfg(time_sampler_cfg)
        params['modality'] = self.modality
        self.time_sampler = model_factory.generate(name, device=device, **params)

        name, params = ConfigLoader.load_model_cfg(space_sampler_cfg)
        params['modality'] = self.modality
        self.space_sampler = model_factory.generate(name, device=device, **params)

        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        params.update({
            'feature_dim': self.light_model.feature_dim,  # TODO: make it from space sampler
            'modality': self.modality,
            'num_class': self.num_class,
            'num_segments': self.num_segments,
            'hallu_dim': self.attention_dim,
            'dropout': self.dropout,
        })
        self.actreg_model = model_factory.generate(name, device=device, **params)

        # Loss for belief propagation
        if hallu_loss == 'mse':
            self.belief_criterion = torch.nn.MSELoss()
        elif hallu_loss == 'ssim':
            self.belief_criterion = SSIM(window_size=3, channel=self.attention_dim[0])
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward function for training"""
        # Light weight model and earlier attention
        x = self.light_model(x)
        attn = self.light_model.rgb.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        attn = attn.view([-1, self.num_segments] + list(attn.shape[1:]))
        self._attn = attn

        # Time sampler
        x = self.time_sampler(x, self.heavy_model)

        # Space sampler
        x = self.space_sampler(x)

        # Action recognition with RNN and hallucination
        output, hallu = self.actreg_model(x)
        self._hallu = hallu

        return output, self.compare_belief().unsqueeze(dim=0)

    def compare_belief(self):
        """Compare between attention and hallucination. Do NOT call directly.

        If using multiple GPUs, self._hallu and self._attn will not be available
        """
        assert hasattr(self, '_attn') and hasattr(self, '_hallu'), \
            'Attributes are not found'
        assert self._attn.shape == self._hallu.shape, 'Mismatching shapes'
        assert torch.all(self._attn >= 0) and torch.all(self._hallu >= 0)

        # TODO: sigmoid, tanh??
        # Change the range??

        # Get attention of future frames
        attn_future = self._attn[:, 1:]

        # Get hallucination from current frames (for future frames)
        hallu_current = self._hallu[:, :-1]

        # Compare belief
        if self.hallu_loss == 'mse':
            loss_belief = self.belief_criterion(hallu_current, attn_future)
        elif self.hallu_loss == 'ssim':
            # Reshape (B,T,C,H,W) --> (B*T,C,H,W) to compare individual images
            # Reverse the sign to maximize SSIM loss
            loss_belief = -self.belief_criterion(
                hallu_current.reshape([-1] + self.attention_dim),
                attn_future.reshape([-1] + self.attention_dim),
            )
        return loss_belief

    def freeze_fn(self, freeze_mode):
        self.light_model.freeze_fn(freeze_mode)

    @property
    def input_mean(self):
        return self.light_model.input_mean

    @property
    def input_std(self):
        return self.light_model.input_std

    @property
    def crop_size(self):
        return self.light_model.input_size

    @property
    def scale_size(self):
        return self.light_model.scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        # if len(self.modality) > 1:
        #     param_groups = []
        #     try:
        #         param_groups.append({'params': filter(lambda p: p.requires_grad, self.light_model.rgb.parameters())})
        #     except AttributeError:
        #         pass

        #     try:
        #         param_groups.append({'params': filter(lambda p: p.requires_grad, self.light_model.flow.parameters()), 'lr': 0.001})
        #     except AttributeError:
        #         pass

        #     try:
        #         param_groups.append({'params': filter(lambda p: p.requires_grad, self.light_model.spec.parameters())})
        #     except AttributeError:
        #         pass

        #     param_groups += [
        #         {'params': filter(lambda p: p.requires_grad, self.heavy_model.parameters())},
        #         {'params': filter(lambda p: p.requires_grad, self.time_sampler.parameters())},
        #         {'params': filter(lambda p: p.requires_grad, self.space_sampler.parameters())},
        #         {'params': filter(lambda p: p.requires_grad, self.actreg_model.parameters())},
        #     ]
        # else:
        #    param_groups = filter(lambda p: p.requires_grad, self.parameters())
        param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
