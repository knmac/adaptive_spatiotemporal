import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader


class PipelineSimple(BaseModel):
    """Simple pipeline with spatial and temporal sampler"""
    def __init__(self, device, model_factory, num_class, num_segments, modality,
                 new_length, dropout, midfusion, consensus_type,
                 light_model_cfg, heavy_model_cfg, time_sampler_cfg,
                 space_sampler_cfg, actreg_model_cfg, using_cupy=True):
        super(PipelineSimple, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.dropout = dropout
        self.midfusion = midfusion
        self.consensus_type = consensus_type
        self.using_cupy = using_cupy

        assert midfusion == 'concat', 'Only support concat for now'

        # Generate models
        name, params = ConfigLoader.load_model_cfg(light_model_cfg)
        params.update({
            'new_length': self.new_length,
            'dropout': self.dropout,
            'num_class': self.num_class,
            'num_segments': self.num_segments,
            'modality': self.modality,
            'midfusion': self.midfusion,
            'consensus_type': self.consensus_type,
            'using_cupy': self.using_cupy,
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
            'dropout': self.dropout,
        })
        self.actreg_model = model_factory.generate(name, device=device, **params)

    def forward(self, input):
        # x, past_belief = input[0], input[1]
        x = input

        x = self.light_model(x)
        x = self.time_sampler(x, self.heavy_model)
        x = self.space_sampler(x)
        output = self.actreg_model(x)

        return output

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
        if len(self.modality) > 1:
            param_groups = []
            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.light_model.rgb.parameters())})
            except AttributeError:
                pass

            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.light_model.flow.parameters()), 'lr': 0.001})
            except AttributeError:
                pass

            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.light_model.spec.parameters())})
            except AttributeError:
                pass

            param_groups += [
                {'params': filter(lambda p: p.requires_grad, self.heavy_model.parameters())},
                {'params': filter(lambda p: p.requires_grad, self.time_sampler.parameters())},
                {'params': filter(lambda p: p.requires_grad, self.space_sampler.parameters())},
                {'params': filter(lambda p: p.requires_grad, self.actreg_model.parameters())},
            ]
        else:
            param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
