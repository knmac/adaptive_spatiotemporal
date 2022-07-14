"""Pipeline version 5

Has spatial sampler.

Train only action recognition (others are frozen) but the inputs are from
multiple resolutions
"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader
from tools.complexity import get_model_complexity_info
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class Pipeline5(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim, dropout,
                 high_feat_model_cfg, low_feat_model_cfg, spatial_sampler_cfg,
                 actreg_model_cfg, feat_process_type, using_cupy, reduce_dim=-1,
                 ignore_lowres=False, full_weights=None):
        super(Pipeline5, self).__init__(device)

        # Turn off cudnn benchmark because of different input size
        # This is only effective whenever pipeline5 is used
        # torch.backends.cudnn.benchmark = False

        # Save the input arguments
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.feat_process_type = feat_process_type  # [reduce, add, cat]
        self.using_cupy = using_cupy
        self.ignore_lowres = ignore_lowres  # ignore low res feature and use only sampled high res
        if ignore_lowres:
            assert feat_process_type == 'cat'  # only works with cat

        # Generate feature extraction models for low resolutions
        name, params = ConfigLoader.load_model_cfg(low_feat_model_cfg)
        assert params['new_input_size'] in [112, 64], \
            'Only support low resolutions of 112 or 64 for now'
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
            'using_cupy': self.using_cupy,
        })
        self.low_feat_model = model_factory.generate(name, device=device, **params)

        # Pivot modality to extract attention
        if 'RGB' in self.modality:
            self._pivot_mod_name = 'RGB'
            self._pivot_mod_fn = self.low_feat_model.rgb
        else:
            raise NotImplementedError

        # Generate feature extraction models for high resolutions
        name, params = ConfigLoader.load_model_cfg(high_feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': [self._pivot_mod_name],  # Remove spec because low_model already has it
            'using_cupy': self.using_cupy,
        })
        self.high_feat_model = model_factory.generate(name, device=device, **params)

        # Generate spatial sampler
        name, params = ConfigLoader.load_model_cfg(spatial_sampler_cfg)
        self.spatial_sampler = model_factory.generate(name, **params)

        # Feature processing functions
        if self.feat_process_type == 'reduce':
            # Reduce dimension of each feature
            self.reduce_dim = reduce_dim

            # FC layers to reduce feature dimension
            self.fc_reduce_low = nn.Linear(
                in_features=self.low_feat_model.feature_dim,
                out_features=reduce_dim,
            ).to(device)
            self.fc_reduce_high = nn.Linear(
                in_features=self.high_feat_model.feature_dim,
                out_features=reduce_dim,
            ).to(device)
            self.fc_reduce_spec = nn.Linear(
                in_features=self.low_feat_model.feature_dim,
                out_features=reduce_dim,
            ).to(device)
            self.relu = nn.ReLU(inplace=True)

            real_dim = self.fc_reduce_low.out_features + \
                self.fc_reduce_spec.out_features + \
                self.fc_reduce_high.out_features*self.spatial_sampler.top_k
        elif self.feat_process_type == 'add':
            # Combine the top k features from high rgb by adding,
            # Make sure the feature dimensions are the same
            assert self.low_feat_model.feature_dim == self.high_feat_model.feature_dim, \
                'Feature dimensions must be the same to add'
            real_dim = self.low_feat_model.feature_dim
        elif self.feat_process_type == 'cat':
            if self.ignore_lowres:
                real_dim = self.low_feat_model.feature_dim * (len(modality)-1) + \
                    self.high_feat_model.feature_dim * self.spatial_sampler.top_k
            else:
                real_dim = self.low_feat_model.feature_dim * len(modality) + \
                    self.high_feat_model.feature_dim * self.spatial_sampler.top_k
        else:
            raise NotImplementedError

        # Generate action recognition model
        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        assert name in ['ActregGRU2', 'ActregGRU3', 'ActregFc'], \
            'Unsupported model: {}'.format(name)
        if name == 'ActregGRU2':
            params.update({
                'feature_dim': 0,  # Use `real_dim` instead
                'extra_dim': real_dim,
                'modality': self.modality,
                'num_class': self.num_class,
                'dropout': self.dropout,
            })
        elif name == 'ActregGRU3':
            params.update({
                'dim_global': len(self.modality)*self.low_feat_model.feature_dim,
                'dim_local': self.spatial_sampler.top_k*self.high_feat_model.feature_dim,
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

        # Overwrite with the full_weights if given
        if full_weights is not None:
            self.load_model(full_weights)

        self.compute_model_complexity()

    def compute_model_complexity(self):
        opts = {'as_strings': False, 'print_per_layer_stat': False}

        # RGB - low res -------------------------------------------------------
        rgb_low_indim = self.low_feat_model.input_size[self._pivot_mod_name]
        if self._pivot_mod_name == 'RGB':
            rgb_low_flops, rgb_low_params = get_model_complexity_info(
                self.low_feat_model.rgb, (3, rgb_low_indim, rgb_low_indim), **opts)
            flops_dict, param_dict = MiscUtils.collect_flops(self.low_feat_model.rgb)

        rgb_low_flops *= 1e-9
        logger.info('%s low (%03d):      GFLOPS=%.04f' %
                    (self._pivot_mod_name, rgb_low_indim, rgb_low_flops))

        # RGB - cropped high res ----------------------------------------------
        assert self.spatial_sampler.min_b_size == self.spatial_sampler.max_b_size
        sampling_size = self.spatial_sampler.max_b_size
        if self._pivot_mod_name == 'RGB':
            rgb_high_flops, rgb_high_params = get_model_complexity_info(
                self.high_feat_model.rgb, (3, sampling_size, sampling_size), **opts)
        rgb_high_flops = rgb_high_flops * self.spatial_sampler.top_k * 1e-9
        logger.info('%s high (%03d, %dx): GFLOPS=%.04f' %
                    (self._pivot_mod_name, sampling_size,
                     self.spatial_sampler.top_k, rgb_high_flops))

        # Spec ----------------------------------------------------------------
        if 'Spec' in self.modality:
            self.low_feat_model.input_size['Spec'] = 256
            spec_indim = self.low_feat_model.input_size['Spec']
            spec_flops, spec_params = get_model_complexity_info(
                self.low_feat_model.spec, (1, spec_indim, spec_indim), **opts)
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
                'rgb_low': rgb_low_flops,
                'rgb_high': rgb_high_flops,
                'spec': spec_flops,
                'actreg': actreg_flops,
            }
        else:
            self.gflops_dict = {
                'rgb_low': rgb_low_flops,
                'rgb_high': rgb_high_flops,
                'actreg': actreg_flops,
            }

        # GFLOPS of the full pipeline
        logger.info('='*33)
        self.gflops_full = sum([v for k, v in self.gflops_dict.items()])
        logger.info('Full pipeline:      GFLOPS=%.4f' % self.gflops_full)

    def _downsample(self, x):
        """Downsample/rescale high resolution image to make low resolution version

        Args:
            x: high resolution image tensor, shape of (B, T*3, H, W)

        Return:
            Low resolution version of x
        """
        high_dim = self.high_feat_model.input_size[self._pivot_mod_name]
        low_dim = self.low_feat_model.input_size[self._pivot_mod_name]
        down_factor = high_dim / low_dim

        if isinstance(down_factor, int):
            return x[:, :, ::down_factor, ::down_factor]
        return F.interpolate(x, size=low_dim, mode='bilinear', align_corners=False)

    def forward(self, x):
        _rgb_high = x[self._pivot_mod_name]
        _rgb_low = self._downsample(_rgb_high)
        if 'Spec' in self.modality:
            _spec = x['Spec']
        batch_size = _rgb_high.shape[0]

        # Extract low resolutions features ------------------------------------
        assert self.low_feat_model.modality == ['RGB', 'Spec'] or \
            self.low_feat_model.modality == ['RGB']
        if 'Spec' in self.modality:
            low_feat, spec_feat = self.low_feat_model(
                {self._pivot_mod_name: _rgb_low, 'Spec': _spec},
                return_concat=False)
        else:
            low_feat = self.low_feat_model(
                {self._pivot_mod_name: _rgb_low},
                return_concat=False)[0]

        # (B*T, C) --> (B, T, C)
        low_feat = low_feat.view([batch_size,
                                  self.num_segments,
                                  self.low_feat_model.feature_dim])
        if 'Spec' in self.modality:
            spec_feat = spec_feat.view([batch_size,
                                        self.num_segments,
                                        self.low_feat_model.feature_dim])

        # Feature processing
        if self.feat_process_type == 'reduce':
            low_feat = self.relu(self.fc_reduce_low(low_feat))
            if 'Spec' in self.modality:
                spec_feat = self.relu(self.fc_reduce_spec(spec_feat))
        elif self.feat_process_type == 'add':
            # Do nothing
            pass
        elif self.feat_process_type == 'cat':
            # Do nothing
            pass

        # Retrieve attention --------------------------------------------------
        attn = self._pivot_mod_fn.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        attn = attn.view([-1, self.num_segments] + list(attn.shape[1:]))
        self._attn = attn

        # Spatial sampler -----------------------------------------------------
        # Compute bboxes -> (B, T, top_k, 4)
        bboxes = self.spatial_sampler.sample_multiple_frames(attn, _rgb_high.shape[-1])

        # (B, T*C, H, W) -> (B, T, C, H, W)
        # self._check(_rgb_high, attn, bboxes)
        n_channels = _rgb_high.shape[1] // self.num_segments
        _rgb_high = _rgb_high.view((-1, self.num_segments, n_channels) + _rgb_high.size()[-2:])

        # Extract regions and feed in high_feat_model
        regions = self.spatial_sampler.get_regions_from_bboxes(_rgb_high, bboxes)
        high_feat = self.high_feat_model(
            {self._pivot_mod_name: torch.cat(regions, dim=0)})
        if self.feat_process_type == 'reduce':
            high_feat = self.relu(self.fc_reduce_high(high_feat))
        high_feat = high_feat.view(
            self.spatial_sampler.top_k, batch_size, self.num_segments, -1)
        high_feat = [item for item in high_feat]

        # Action recognition --------------------------------------------------
        if 'Spec' in self.modality:
            if self.feat_process_type == 'reduce':
                all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=2)
            elif self.feat_process_type == 'add':
                all_feats = low_feat + spec_feat
                for k in range(self.spatial_sampler.top_k):
                    all_feats += high_feat[k]
            elif self.feat_process_type == 'cat':
                if self.ignore_lowres:
                    all_feats = torch.cat([spec_feat] + high_feat, dim=2)
                else:
                    all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=2)
        else:
            if self.feat_process_type == 'reduce':
                all_feats = torch.cat([low_feat] + high_feat, dim=2)
            elif self.feat_process_type == 'add':
                all_feats = low_feat
                for k in range(self.spatial_sampler.top_k):
                    all_feats += high_feat[k]
            elif self.feat_process_type == 'cat':
                if self.ignore_lowres:
                    all_feats = torch.cat(high_feat, dim=2)
                else:
                    all_feats = torch.cat([low_feat] + high_feat, dim=2)

        assert all_feats.ndim == 3

        output, hidden = self.actreg_model(all_feats, hidden=None)

        # Does not need belief_loss because the function compare_belief is not
        # available here
        return output

    def _check(self, img, attn, bboxes, ix=0):
        """Visualize to check the results of spatial sampler. For debugging only
        """
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        matplotlib.use('TkAgg')
        sns.set_style('whitegrid', {'axes.grid': False})

        img = MiscUtils.deprocess_rgb(img[ix], self.num_segments)
        attn = attn[ix].cpu().detach().numpy().mean(axis=1)
        bboxes = bboxes[ix]

        fig, axes = plt.subplots(3, self.num_segments)
        for t in range(self.num_segments):
            axes[0, t].imshow(img[t])
            axes[1, t].imshow(attn[t], vmin=attn.min(), vmax=attn.max())

            frame = np.zeros(img.shape[1:], dtype=np.uint8)
            bbox = bboxes[t]
            for k in range(self.spatial_sampler.top_k):
                frame[bbox[k, 0]:bbox[k, 2], bbox[k, 1]:bbox[k, 3], k] = 255
            axes[2, t].imshow(frame)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def freeze_fn(self, freeze_mode):
        self.low_feat_model.freeze_fn(freeze_mode)
        self.high_feat_model.freeze_fn(freeze_mode)

    @property
    def input_mean(self):
        return self.high_feat_model.input_mean

    @property
    def input_std(self):
        # because low_feat_model has spec std
        return self.low_feat_model.input_std

    @property
    def crop_size(self):
        return self.high_feat_model.input_size

    @property
    def scale_size(self):
        return self.high_feat_model.scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
