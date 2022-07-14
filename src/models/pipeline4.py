"""Pipeline version 4 - Run only action recognition

Allow feat_process_type of ['cat', 'add']
"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader
from tools.complexity import get_model_complexity_info
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class Pipeline4(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, dropout, feat_model_cfg, actreg_model_cfg,
                 feat_process_type='cat', using_cupy=True):
        super(Pipeline4, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.dropout = dropout
        self.feat_process_type = feat_process_type
        self.using_cupy = using_cupy

        # Generate feature extraction model
        name, params = ConfigLoader.load_model_cfg(feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
            'using_cupy': self.using_cupy,
        })
        self.feat_model = model_factory.generate(name, device=device, **params)

        # Generate action recognition model
        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        assert name in ['ActregGRU2', 'ActregFc'], \
            'Unsupported model: {}'.format(name)

        if self.feat_process_type == 'cat':
            real_dim = self.feat_model.feature_dim * len(modality)
        elif self.feat_process_type == 'add':
            real_dim = self.feat_model.feature_dim
        else:
            raise NotImplementedError
        if name == 'ActregGRU2':
            params.update({
                'feature_dim': 0,  # Use `real_dim` instead
                'extra_dim': real_dim,
                'modality': self.modality,
                'num_class': self.num_class,
                'dropout': self.dropout,
            })
        elif name == 'ActregFc':
            params.update({
                'feature_dim': real_dim,
                'modality': self.modality,
                'num_class': self.num_class,
                'dropout': self.dropout,
                'num_segments': self.num_segments,
            })
        self.actreg_model = model_factory.generate(name, device=device, **params)
        self.actreg_model.to(self.device)

        self.compute_model_complexity()

    def compute_model_complexity(self):
        opts = {'as_strings': False, 'print_per_layer_stat': False}

        # Pivot modality
        if 'RGB' in self.modality:
            self._pivot_mod_name = 'RGB'
        else:
            raise NotImplementedError

        # RGB -----------------------------------------------------------------
        rgb_indim = self.feat_model.input_size[self._pivot_mod_name]
        if self._pivot_mod_name == 'RGB':
            rgb_flops, rgb_params = get_model_complexity_info(
                self.feat_model.rgb, (3, rgb_indim, rgb_indim), **opts)
            flops_dict, param_dict = MiscUtils.collect_flops(self.feat_model.rgb)

        rgb_flops *= 1e-9
        logger.info('%s (%03d):          GFLOPS=%.04f' %
                    (self._pivot_mod_name, rgb_indim, rgb_flops))

        # Spec ----------------------------------------------------------------
        if 'Spec' in self.modality:
            self.feat_model.input_size['Spec'] = 256
            spec_indim = self.feat_model.input_size['Spec']
            spec_flops, spec_params = get_model_complexity_info(
                self.feat_model.spec, (1, spec_indim, spec_indim), **opts)
            spec_flops *= 1e-9
            logger.info('Spec (%03d):         GFLOPS=%.04f' % (spec_indim, spec_flops))

        # Actreg --------------------------------------------------------------
        if type(self.actreg_model).__name__ == 'ActregFc':
            # NOTE: Force 1 frame to compute complexity for fc model
            self.actreg_model.num_segments = 1
            actreg_flops, actreg_params = get_model_complexity_info(
                self.actreg_model, (1, self.actreg_model._input_dim), **opts)
            self.actreg_model.num_segments = self.num_segments
        else:
            actreg_flops, actreg_params = get_model_complexity_info(
                self.actreg_model, (1, self.actreg_model._input_dim), **opts)
        actreg_flops *= 1e-9
        logger.info('Actreg:             GFLOPS=%.4f' % actreg_flops)

        if 'Spec' in self.modality:
            self.gflops_dict = {
                'rgb': rgb_flops,
                'spec': spec_flops,
                'actreg': actreg_flops,
            }
        else:
            self.gflops_dict = {
                'rgb': rgb_flops,
                'actreg': actreg_flops,
            }

        # GFLOPS of the full pipeline
        logger.info('='*33)
        self.gflops_full = sum([v for k, v in self.gflops_dict.items()])
        logger.info('Full pipeline:      GFLOPS=%.4f' % self.gflops_full)

    def forward(self, x):
        # Extract features
        batch_size = x[self.modality[0]].shape[0]
        if self.feat_process_type == 'cat':
            x = self.feat_model(x)
        elif self.feat_process_type == 'add':
            x_lst = self.feat_model(x, return_concat=False)
            x = torch.stack(x_lst).sum(dim=0)

        # (B*T, C) --> (B, T, C)
        target_dim = x.shape[-1]
        x = x.view([batch_size, self.num_segments, target_dim])

        # Action recognition
        hidden = None
        output, hidden = self.actreg_model(x, hidden)

        # Does not need belief_loss because the function compare_belief is not
        # available here
        return output

    def freeze_fn(self, freeze_mode):
        self.feat_model.freeze_fn(freeze_mode)

    @property
    def input_mean(self):
        return self.feat_model.input_mean

    @property
    def input_std(self):
        return self.feat_model.input_std

    @property
    def crop_size(self):
        return self.feat_model.input_size

    @property
    def scale_size(self):
        return self.feat_model.scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
