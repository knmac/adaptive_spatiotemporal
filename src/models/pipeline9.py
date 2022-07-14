"""Pipeline9 - Full pipeline with all components

Similar to pipeline8, with more optimization
"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from torch.nn import functional as F

from .base_model import BaseModel
from tools.complexity import get_model_complexity_info
from src.utils.load_cfg import ConfigLoader
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class Pipeline9(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim, dropout,
                 low_feat_model_cfg, high_feat_model_cfg, hallu_model_cfg,
                 actreg_model_cfg, spatial_sampler_cfg, temporal_sampler_cfg,
                 hallu_pretrained_weights, actreg_pretrained_weights,
                 feat_process_type, freeze_hallu, freeze_actreg, temperature,
                 temperature_exp_decay_factor, eff_loss_weights, usage_loss_weights,
                 using_cupy, full_weights=None):
        super(Pipeline9, self).__init__(device)

        # Only support layer3_0 and layer2_3 for now
        assert attention_layer in [['layer3', '0'], ['layer2', '3']]

        # Save the input arguments
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.feat_process_type = feat_process_type  # [add, cat]
        self.using_cupy = using_cupy
        self.eff_loss_weights = eff_loss_weights
        self.usage_loss_weights = usage_loss_weights

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

        # Generate temporal sampler
        self.init_temperature = temperature
        self.temperature = temperature
        self.temperature_exp_decay_factor = temperature_exp_decay_factor
        name, params = ConfigLoader.load_model_cfg(temporal_sampler_cfg)
        params.update({
            'attention_dim': self.attention_dim,
        })
        self.temporal_sampler = model_factory.generate(name, device=device, **params)
        self.temporal_sampler.to(self.device)

        # Generate spatial sampler
        name, params = ConfigLoader.load_model_cfg(spatial_sampler_cfg)
        self.spatial_sampler = model_factory.generate(name, **params)

        # Feature processing functions
        if self.feat_process_type == 'add':
            # Combine the top k features from high rgb by adding,
            # Make sure the feature dimensions are the same
            assert self.low_feat_model.feature_dim == self.high_feat_model.feature_dim, \
                'Feature dimensions must be the same to add'
            real_dim = self.low_feat_model.feature_dim
        elif self.feat_process_type == 'cat':
            real_dim = self.low_feat_model.feature_dim * len(modality) + \
                self.high_feat_model.feature_dim * self.spatial_sampler.top_k
        else:
            raise NotImplementedError

        # Generate hallucination model
        name, params = ConfigLoader.load_model_cfg(hallu_model_cfg)
        assert name in ['HalluConvLSTM2']
        params.update({
            'attention_dim': self.attention_dim,
        })
        self.hallu_model = model_factory.generate(name, device=device, **params)
        if hallu_pretrained_weights is not None:
            self.hallu_model.load_model(hallu_pretrained_weights)
        self.hallu_model.to(self.device)

        if freeze_hallu:
            for param in self.hallu_model.parameters():
                param.requires_grad_(False)

        # Generate action recognition model
        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        assert name in ['ActregGRU3', 'ActregGRU2', 'ActregFc'], \
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
        if actreg_pretrained_weights is not None:
            self.actreg_model.load_model(actreg_pretrained_weights)
        self.actreg_model.to(self.device)

        if freeze_actreg:
            for param in self.actreg_model.parameters():
                param.requires_grad_(False)

        # Overwrite with the full_weights if given
        if full_weights is not None:
            self.load_model(full_weights)

        # Compute model complexity
        self.compute_model_complexity()

    def compute_model_complexity(self):
        """Compute the flops of every models
        """
        opts = {'as_strings': False, 'print_per_layer_stat': False}

        # RGB - low res -------------------------------------------------------
        rgb_low_indim = self.low_feat_model.input_size[self._pivot_mod_name]
        if self._pivot_mod_name == 'RGB':
            rgb_low_flops, rgb_low_params = get_model_complexity_info(
                self.low_feat_model.rgb, (3, rgb_low_indim, rgb_low_indim), **opts)
            flops_dict, param_dict = MiscUtils.collect_flops(self.low_feat_model.rgb)

        rgb_low_first_flops, rgb_low_first_params = 0, 0
        for k in flops_dict:
            rgb_low_first_flops += flops_dict[k]
            rgb_low_first_params += param_dict[k]
            if k == '-'.join(self.attention_layer):
                break
        rgb_low_second_flops = rgb_low_flops - rgb_low_first_flops
        # rgb_low_second_params = rgb_low_params - rgb_low_first_params

        rgb_low_flops *= 1e-9
        rgb_low_first_flops *= 1e-9
        rgb_low_second_flops *= 1e-9
        logger.info('%s low (%03d):      GFLOPS=%.04f' %
                    (self._pivot_mod_name, rgb_low_indim, rgb_low_flops))
        logger.info('- 1st half:         GFLOPS=%.04f' % rgb_low_first_flops)
        logger.info('- 2nd half:         GFLOPS=%.04f' % rgb_low_second_flops)

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

        # Hallucination -------------------------------------------------------
        input_dim = tuple([1] + self.attention_dim)
        hallu_flops, hallu_params = get_model_complexity_info(
            self.hallu_model, input_dim, **opts)
        hallu_flops *= 1e-9
        logger.info('Hallucination:      GFLOPS=%.4f' % hallu_flops)

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

        # Time sampler --------------------------------------------------------
        time_sampler_flops, time_sampler_params = get_model_complexity_info(
            self.temporal_sampler, tuple([2]+self.attention_dim), **opts)
        time_sampler_flops *= 1e-9
        logger.info('Time sampler:       GFLOPS=%.4f' % time_sampler_flops)

        if 'Spec' in self.modality:
            self.gflops_dict = {
                'rgb_low_first': rgb_low_first_flops,
                'rgb_low_second': rgb_low_second_flops,
                'rgb_high': rgb_high_flops,
                'spec': spec_flops,
                'hallu': hallu_flops,
                'actreg': actreg_flops,
                'time_sampler': time_sampler_flops,
            }
        else:
            self.gflops_dict = {
                'rgb_low_first': rgb_low_first_flops,
                'rgb_low_second': rgb_low_second_flops,
                'rgb_high': rgb_high_flops,
                'hallu': hallu_flops,
                'actreg': actreg_flops,
                'time_sampler': time_sampler_flops,
            }

        # GFLOPS of the full pipeline -----------------------------------------
        logger.info('='*33)
        self.gflops_full = sum([v for k, v in self.gflops_dict.items()])
        logger.info('Full pipeline:      GFLOPS=%.4f' % self.gflops_full)

        # GFLOPS for only prescanning
        self.gflops_prescan = sum([self.gflops_dict[k] for k in
                                   ['rgb_low_first', 'hallu', 'time_sampler']])
        logger.info('Prescanning:        GFLOPS=%.4f' % self.gflops_prescan)

    def decay_temperature(self, epoch):
        """Decay the temperature for gumbel softmax loss"""
        self.temperature = self.init_temperature * np.exp(self.temperature_exp_decay_factor * epoch)

    def _downsample(self, x):
        """Downsample high resolution image to make low resolution version

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
        return F.interpolate(x, size=low_dim,
                             mode='bilinear', align_corners=False)

    def forward(self, x, get_extra=False):
        # =====================================================================
        # Prepare inputs
        # =====================================================================
        rgb_high = x[self._pivot_mod_name]
        rgb_low = self._downsample(rgb_high)
        if 'Spec' in self.modality:
            spec = x['Spec']
        batch_size = rgb_high.shape[0]

        # (B, T*C, H, W) -> (B, T, C, H, W)
        n_channels = rgb_low.shape[1] // self.num_segments
        rgb_low = rgb_low.view((-1, self.num_segments, n_channels) + rgb_low.size()[-2:])
        rgb_high = rgb_high.view((-1, self.num_segments, n_channels) + rgb_high.size()[-2:])
        if 'Spec' in self.modality:
            spec = spec.view((-1, self.num_segments, 1) + spec.size()[-2:])

        # =====================================================================
        # Extract features by batch
        # =====================================================================
        # Low feat and spec feat
        assert self.low_feat_model.modality == ['RGB', 'Spec'] or \
            self.low_feat_model.modality == ['RGB']
        if 'Spec' in self.modality:
            low_feat, spec_feat = self.low_feat_model(
                {self._pivot_mod_name: rgb_low, 'Spec': spec},
                return_concat=False)
        else:
            low_feat = self.low_feat_model(
                {self._pivot_mod_name: rgb_low},
                return_concat=False)[0]

        # (B*T, C) --> (B, T, C)
        low_feat = low_feat.view([batch_size, self.num_segments,
                                  self.low_feat_model.feature_dim])
        if 'Spec' in self.modality:
            spec_feat = spec_feat.view([batch_size, self.num_segments,
                                        self.low_feat_model.feature_dim])

        # Retrieve attention from the 1st half
        attn = self._pivot_mod_fn.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        attn = attn.view([-1, self.num_segments] + list(attn.shape[1:]))

        # =====================================================================
        # Temporal sampling
        # =====================================================================
        r_all = self.temporal_sampler.sample_multiple_frames(
            attn, self.hallu_model, self.temperature)

        # =====================================================================
        # Spatial sampling
        # =====================================================================
        if self.spatial_sampler.top_k != 0:
            # Extract bboxes and corresponding regions
            bboxes = self.spatial_sampler.sample_multiple_frames(
                attn, rgb_high.shape[-1], reorder_vid=False)
            regions = self.spatial_sampler.get_regions_from_bboxes(rgb_high, bboxes)

            # Extract high feat
            high_feat = self.high_feat_model(
                {self._pivot_mod_name: torch.cat(regions, dim=0)})
            high_feat = high_feat.view(
                self.spatial_sampler.top_k, batch_size, self.num_segments, -1)
            high_feat = [item for item in high_feat]

            # Sort sampled bboxes to permute high_feat
            order = np.tile(np.arange(self.spatial_sampler.top_k),
                            (batch_size, self.num_segments, 1))
            for batch_i in range(batch_size):
                indices = torch.where(r_all[batch_i, :, 0] == 1)[0].cpu().detach().numpy()
                bboxes_samp = bboxes[batch_i, indices]
                _, reorder = self.spatial_sampler._sort_bboxes_dijkstra(bboxes_samp)
                order[batch_i, indices, :] = reorder

        # =====================================================================
        # Classification
        # =====================================================================
        has_multihead = type(self.actreg_model).__name__ in ['ActregGRU3']
        has_multiclass = isinstance(self.num_class, list)
        old_pred, old_pred_mem = None, None
        pred_all = []

        for t in range(self.num_segments):
            # Prepare features
            if self.feat_process_type == 'add':
                if 'Spec' in self.modality:
                    all_feats = low_feat[:, t] + spec_feat[:, t]
                else:
                    all_feats = low_feat[:, t]
                for k in range(self.spatial_sampler.top_k):
                    all_feats += high_feat[k][:, t]
            elif self.feat_process_type == 'cat':
                high_feat_t = []
                # Reorder high-res features
                for k in range(self.spatial_sampler.top_k):
                    high_feat_t_b = []
                    for batch_i in range(batch_size):
                        idx = order[batch_i, t, k]
                        high_feat_t_b.append(high_feat[idx][batch_i, t])
                    high_feat_t.append(torch.stack(high_feat_t_b, dim=0))
                if 'Spec' in self.modality:
                    all_feats = torch.cat([low_feat[:, t], spec_feat[:, t]] + high_feat_t, dim=1)
                else:
                    all_feats = torch.cat([low_feat[:, t]] + high_feat_t, dim=1)
            assert all_feats.ndim == 2
            all_feats = all_feats.unsqueeze(dim=1)

            # Feed features to classifier
            # NOTE: GRU memory has shape (layer, batch, dim)
            pred, pred_mem = self.actreg_model(all_feats, old_pred_mem)

            # Update states by batch, only update based on r_all
            if old_pred_mem is not None:
                # Only run full pipeline if not skipping (r_all[..., 0] = 1)
                take_bool = (r_all[:, t, 0] < 0.5).unsqueeze(dim=-1)
                # take_old = torch.tensor(take_bool, dtype=torch.float).to(r_all.device)
                # take_curr = torch.tensor(~take_bool, dtype=torch.float).to(r_all.device)
                take_old = take_bool.to(r_all.device, torch.float)
                take_curr = 1.0 - take_old

                if not has_multihead:
                    if not has_multiclass:  # non-epic dataset
                        pred = (old_pred * take_old) + (pred * take_curr)
                    else:
                        pred = ((old_pred[0] * take_old) + (pred[0] * take_curr),
                                (old_pred[1] * take_old) + (pred[1] * take_curr))
                    pred_mem = (old_pred_mem * take_old.unsqueeze(0)) + (pred_mem * take_curr.unsqueeze(0))
                else:
                    assert self.actreg_model._n_heads == 3
                    if not has_multiclass:  # non-epic dataset
                        pred = (
                            # Head 1
                            (old_pred[0] * take_old) + (pred[0] * take_curr),
                            (old_pred[1] * take_old) + (pred[1] * take_curr),
                            # Head 2
                            (old_pred[2] * take_old) + (pred[2] * take_curr),
                            (old_pred[3] * take_old) + (pred[3] * take_curr),
                            # Head 3
                            (old_pred[4] * take_old) + (pred[4] * take_curr),
                            (old_pred[5] * take_old) + (pred[5] * take_curr),
                        )
                    else:
                        pred = (
                            # Head 1
                            (
                                (old_pred[0][0] * take_old) + (pred[0][0] * take_curr),
                                (old_pred[0][1] * take_old) + (pred[0][1] * take_curr),
                            ),
                            (old_pred[1] * take_old) + (pred[1] * take_curr),
                            # Head 2
                            (
                                (old_pred[2][0] * take_old) + (pred[2][0] * take_curr),
                                (old_pred[2][1] * take_old) + (pred[2][1] * take_curr),
                            ),
                            (old_pred[3] * take_old) + (pred[3] * take_curr),
                            # Head 3
                            (
                                (old_pred[4][0] * take_old) + (pred[4][0] * take_curr),
                                (old_pred[4][1] * take_old) + (pred[4][1] * take_curr),
                            ),
                            (old_pred[5] * take_old) + (pred[5] * take_curr),
                        )
                    pred_mem = (
                        (old_pred_mem[0] * take_old.unsqueeze(0)) + (pred_mem[0] * take_curr.unsqueeze(0)),
                        (old_pred_mem[1] * take_old.unsqueeze(0)) + (pred_mem[1] * take_curr.unsqueeze(0)),
                        (old_pred_mem[2] * take_old.unsqueeze(0)) + (pred_mem[2] * take_curr.unsqueeze(0)),
                    )

            old_pred = pred
            old_pred_mem = pred_mem
            pred_all.append(pred)

        if not has_multihead:
            if not has_multiclass:  # non-epic dataset
                pred_all = torch.stack([pred for pred in pred_all], dim=1)
            else:
                pred_all = (torch.stack([pred[0] for pred in pred_all], dim=1),
                            torch.stack([pred[1] for pred in pred_all], dim=1))
        else:
            if not has_multiclass:  # non-epic dataset
                pred_all = (
                    # Head 1
                    torch.stack([pred[0] for pred in pred_all], dim=1),
                    torch.stack([pred[1] for pred in pred_all], dim=1),
                    # Head 2
                    torch.stack([pred[2] for pred in pred_all], dim=1),
                    torch.stack([pred[3] for pred in pred_all], dim=1),
                    # Head 3
                    torch.stack([pred[4] for pred in pred_all], dim=1),
                    torch.stack([pred[5] for pred in pred_all], dim=1),
                )
            else:
                pred_all = (
                    # Head 1
                    (
                        torch.stack([pred[0][0] for pred in pred_all], dim=1),
                        torch.stack([pred[0][1] for pred in pred_all], dim=1),
                    ),
                    torch.stack([pred[1] for pred in pred_all], dim=1),
                    # Head 2
                    (
                        torch.stack([pred[2][0] for pred in pred_all], dim=1),
                        torch.stack([pred[2][1] for pred in pred_all], dim=1),
                    ),
                    torch.stack([pred[3] for pred in pred_all], dim=1),
                    # Head 3
                    (
                        torch.stack([pred[4][0] for pred in pred_all], dim=1),
                        torch.stack([pred[4][1] for pred in pred_all], dim=1),
                    ),
                    torch.stack([pred[5] for pred in pred_all], dim=1),
                )

        # =====================================================================
        # Combine logits
        # =====================================================================
        r_tensor = r_all[:, :, 0].unsqueeze(dim=-1)
        t_tensor = r_tensor.sum(dim=[1, 2]).unsqueeze(dim=-1)
        if not has_multihead:
            if not has_multiclass:  # non-epic dataset
                pred_all = torch.sum(pred_all * r_tensor, dim=1) / t_tensor
            else:
                pred_all = (torch.sum(pred_all[0] * r_tensor, dim=1) / t_tensor,
                            torch.sum(pred_all[1] * r_tensor, dim=1) / t_tensor)
        else:
            if not has_multiclass:  # non-epic dataset
                pred_all = (
                    # Head 1
                    torch.sum(pred_all[0] * r_tensor, dim=1) / t_tensor,
                    pred_all[1].mean(dim=1),
                    # Head 2
                    torch.sum(pred_all[2] * r_tensor, dim=1) / t_tensor,
                    pred_all[3].mean(dim=1),
                    # Head 3
                    torch.sum(pred_all[4] * r_tensor, dim=1) / t_tensor,
                    pred_all[5].mean(dim=1),
                )
            else:
                pred_all = (
                    # Head 1
                    (
                        torch.sum(pred_all[0][0] * r_tensor, dim=1) / t_tensor,
                        torch.sum(pred_all[0][1] * r_tensor, dim=1) / t_tensor,
                    ),
                    pred_all[1].mean(dim=1),
                    # Head 2
                    (
                        torch.sum(pred_all[2][0] * r_tensor, dim=1) / t_tensor,
                        torch.sum(pred_all[2][1] * r_tensor, dim=1) / t_tensor,
                    ),
                    pred_all[3].mean(dim=1),
                    # Head 3
                    (
                        torch.sum(pred_all[4][0] * r_tensor, dim=1) / t_tensor,
                        torch.sum(pred_all[4][1] * r_tensor, dim=1) / t_tensor,
                    ),
                    pred_all[5].mean(dim=1),
                )

        # =====================================================================
        # Compute efficiency loss
        # =====================================================================
        loss_eff, loss_usage, gflops_lst = self.compute_efficiency_loss(r_all)
        gflops_lst = torch.tensor(gflops_lst).to(loss_eff.device)

        if get_extra:
            extra_outputs = {'r': r_all}
            return pred_all, loss_eff, gflops_lst, extra_outputs
        return pred_all, loss_eff, loss_usage, gflops_lst

    def compute_efficiency_loss(self, r_all):
        """Compute the efficient loss based on sampling and models FLOPS

        Args:
            r: sampling vector of shape (N, T, max_frames_skip+1)

        Return:
            loss_eff: efficiency loss of shape (N,)
        """
        batch_size = r_all.shape[0]
        loss_eff = torch.zeros([batch_size]).to(self.device)
        gflops_lst = np.zeros([batch_size, self.num_segments])

        for i in range(batch_size):
            t = 0
            while t < self.num_segments:
                skip = r_all[i, t].argmax().item()
                if skip == 0:
                    loss_eff[i] += r_all[i, t].sum() * self.gflops_full
                    gflops_lst[i, t] = self.gflops_full
                    t += 1
                else:
                    loss_eff[i] += r_all[i, t].sum() + self.gflops_prescan
                    gflops_lst[i, t] = self.gflops_prescan
                    t += skip
        loss_eff = loss_eff.mean() * self.eff_loss_weights

        # Usage loss
        loss_usage = r_all.mean(dim=[0, 1])
        loss_usage = torch.norm(loss_usage - loss_usage.mean()) * self.usage_loss_weights

        # loss_eff += loss_usage
        return loss_eff, loss_usage, gflops_lst

    def freeze_fn(self, freeze_mode):
        self.low_feat_model.freeze_fn(freeze_mode)
        self.high_feat_model.freeze_fn(freeze_mode)
        # self.hallu_model.freeze_fn(freeze_mode)
        # self.actreg_model.freeze_fn(freeze_mode)

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
