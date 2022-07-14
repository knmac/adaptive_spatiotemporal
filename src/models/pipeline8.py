"""Pipeline8 - Full pipeline with all components

Similar to pipeline7, but allow multi-head skipping and flops loss
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from torch.nn import functional as F

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader
from tools.complexity import get_model_complexity_info
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class Pipeline8(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim, dropout,
                 low_feat_model_cfg, high_feat_model_cfg, hallu_model_cfg,
                 actreg_model_cfg, spatial_sampler_cfg, temporal_sampler_cfg,
                 hallu_pretrained_weights, actreg_pretrained_weights,
                 feat_process_type, freeze_hallu, freeze_actreg, temperature,
                 temperature_exp_decay_factor, eff_loss_weights, using_cupy,
                 temporal_sampler_lr=None, full_weights=None):
        super(Pipeline8, self).__init__(device)

        # Turn off cudnn benchmark because of different input size
        # torch.backends.cudnn.benchmark = False

        # Only support layer3_0 and layer2_3 for now
        # Affecting the implementation of 1st and 2nd half splitting
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
        self.temporal_sampler_lr = temporal_sampler_lr  # Overwrite default lr

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
        actreg_flops, actreg_params = get_model_complexity_info(
            self.actreg_model, (1, self.actreg_model._input_dim), **opts)
        actreg_flops *= 1e-9
        logger.info('Actreg:             GFLOPS=%.4f' % actreg_flops)

        # Time sampler --------------------------------------------------------
        time_sampler_flops, time_sampler_params = get_model_complexity_info(
            self.temporal_sampler, tuple([2]+self.attention_dim), **opts)
        time_sampler_flops *= 1e-9
        logger.info('Time sampler:       GFLOPS=%.4f' % time_sampler_flops)

        self.gflops_dict = {
            'rgb_low_first': rgb_low_first_flops,
            'rgb_low_second': rgb_low_second_flops,
            'rgb_high': rgb_high_flops,
            'spec': spec_flops,
            'hallu': hallu_flops,
            'actreg': actreg_flops,
            'time_sampler': time_sampler_flops,
        }

        # GFLOPS of the full pipeline
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

    def forward(self, x, output_mode='avg_non_skip', get_extra=False):
        """Forwad a sequence of frame
        """
        assert output_mode in ['avg_all', 'avg_non_skip', 'raw']
        rgb_high = x[self._pivot_mod_name]
        rgb_low = self._downsample(rgb_high)
        spec = x['Spec']
        batch_size = rgb_high.shape[0]

        # (B, T*C, H, W) -> (B, T, C, H, W)
        n_channels = rgb_low.shape[1] // self.num_segments
        rgb_low = rgb_low.view((-1, self.num_segments, n_channels) + rgb_low.size()[-2:])
        rgb_high = rgb_high.view((-1, self.num_segments, n_channels) + rgb_high.size()[-2:])
        spec = spec.view((-1, self.num_segments, 1) + spec.size()[-2:])

        # =====================================================================
        # Extract features by batch (skipped frames won't get counted)
        # =====================================================================
        # First half of feature low rgb feat --> (B, T, C, H, W)
        prescan_feat = self.first_half_forward(
            rgb_low.view((-1, n_channels) + rgb_low.size()[-2:]), self._pivot_mod_fn
        )
        prescan_feat = prescan_feat.view((-1, self.num_segments) + prescan_feat.size()[-3:])

        # Get attention of all frames --> (B, T, C, H, W)
        attn = self._pivot_mod_fn.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        assert attn is not None, 'Fail to retrieve attention'
        attn = attn.view((-1, self.num_segments) + attn.size()[-3:])

        # Second half of RGB low
        low_feat = self.second_half_forward(
            prescan_feat.view((-1,)+prescan_feat.size()[-3:]), self._pivot_mod_fn)
        low_feat = low_feat.view(-1, self.num_segments, low_feat.shape[-1])

        # Spec
        spec_feat = self.low_feat_model.spec(spec.view((-1,)+spec.size()[-3:]))
        spec_feat = spec_feat.view(-1, self.num_segments, spec_feat.shape[-1])

        # =====================================================================
        # Process each sample separately because skipping is non-uniform across
        # batch domain
        # =====================================================================
        has_multihead = type(self.actreg_model).__name__ in ['ActregGRU3']
        outputs, extra_outputs = [], []
        for i in range(batch_size):
            # Reset samplers at the start of the sequence
            self.temporal_sampler.reset()
            self.spatial_sampler.reset()
            all_skip, all_ssim, all_time, all_r = [], [], [], []
            if not has_multihead:
                all_pred_verb, all_pred_noun = [], []
            else:
                assert self.actreg_model._n_heads == 3
                all_pred_verb_0, all_pred_noun_0 = [], []
                all_pred_verb_1, all_pred_noun_1 = [], []
                all_pred_verb_2, all_pred_noun_2 = [], []
                head_w0, head_w1, head_w2 = [], [], []

            # Warm up using the first frame -----------------------------------
            st = time.time()
            result_dict = self.warmup(
                low_feat[i, 0].unsqueeze(0), attn[i, 0].unsqueeze(0),
                spec_feat[i, 0].unsqueeze(0), rgb_high[i, 0].unsqueeze(0),
            )
            actreg_mem = result_dict['actreg_mem']
            hallu_mem = result_dict['hallu_mem']
            pred = result_dict['pred']
            hallu = result_dict['hallu']
            remaining_skips = result_dict['remaining_skips']

            all_time.append(time.time()-st)
            all_skip.append(result_dict['skipped'])
            all_ssim.append(result_dict['ssim'])
            all_r.append(result_dict['r_t'])
            if not has_multihead:
                all_pred_verb.append(pred[0].unsqueeze(dim=1))
                all_pred_noun.append(pred[1].unsqueeze(dim=1))
            else:
                all_pred_verb_0.append(pred[0][0].unsqueeze(dim=1))
                all_pred_noun_0.append(pred[0][1].unsqueeze(dim=1))
                head_w0.append(pred[1])

                all_pred_verb_1.append(pred[2][0].unsqueeze(dim=1))
                all_pred_noun_1.append(pred[2][1].unsqueeze(dim=1))
                head_w1.append(pred[3])

                all_pred_verb_2.append(pred[4][0].unsqueeze(dim=1))
                all_pred_noun_2.append(pred[4][1].unsqueeze(dim=1))
                head_w2.append(pred[5])

            # Forward frame by frame ------------------------------------------
            for t in range(1, self.num_segments):
                st = time.time()
                result_dict = self.forward_frame(
                    low_feat[i, t].unsqueeze(0), attn[i, t].unsqueeze(0),
                    spec_feat[i, t].unsqueeze(0), rgb_high[i, t].unsqueeze(0),
                    old_pred=pred, old_hallu=hallu,
                    actreg_mem=actreg_mem, hallu_mem=hallu_mem,
                    remaining_skips=remaining_skips,
                )

                # Update states
                actreg_mem = result_dict['actreg_mem']
                hallu_mem = result_dict['hallu_mem']
                pred = result_dict['pred']
                hallu = result_dict['hallu']
                remaining_skips = result_dict['remaining_skips']

                # Collect results
                all_time.append(time.time()-st)
                all_skip.append(result_dict['skipped'])
                all_ssim.append(result_dict['ssim'])
                all_r.append(result_dict['r_t'])
                if not has_multihead:
                    all_pred_verb.append(pred[0].unsqueeze(dim=1))
                    all_pred_noun.append(pred[1].unsqueeze(dim=1))
                else:
                    all_pred_verb_0.append(pred[0][0].unsqueeze(dim=1))
                    all_pred_noun_0.append(pred[0][1].unsqueeze(dim=1))
                    head_w0.append(pred[1])

                    all_pred_verb_1.append(pred[2][0].unsqueeze(dim=1))
                    all_pred_noun_1.append(pred[2][1].unsqueeze(dim=1))
                    head_w1.append(pred[3])

                    all_pred_verb_2.append(pred[4][0].unsqueeze(dim=1))
                    all_pred_noun_2.append(pred[4][1].unsqueeze(dim=1))
                    head_w2.append(pred[5])

            # Prepare outputs -------------------------------------------------
            all_ssim = torch.cat(all_ssim, dim=1)
            all_r = torch.cat(all_r, dim=0).unsqueeze(0)
            if not has_multihead:
                all_pred_verb = torch.cat(all_pred_verb, dim=1)
                all_pred_noun = torch.cat(all_pred_noun, dim=1)
                if output_mode == 'avg_all':
                    output = (all_pred_verb.mean(dim=1), all_pred_noun.mean(dim=1))
                elif output_mode == 'avg_non_skip':
                    foo, bar, cnt = 0, 0, 0
                    for t in range(self.num_segments):
                        if not all_skip[t]:
                            foo += all_pred_verb[:, t, :]
                            bar += all_pred_noun[:, t, :]
                            cnt += 1
                    output = (foo/cnt, bar/cnt)
                elif output_mode == 'raw':
                    output = (all_pred_verb, all_pred_noun)
                else:
                    raise NotImplementedError
            else:
                assert output_mode == 'avg_non_skip', NotADirectoryError
                all_pred_verb_0 = torch.cat(all_pred_verb_0, dim=1)
                all_pred_noun_0 = torch.cat(all_pred_noun_0, dim=1)
                head_w0 = torch.cat(head_w0, dim=1)

                all_pred_verb_1 = torch.cat(all_pred_verb_1, dim=1)
                all_pred_noun_1 = torch.cat(all_pred_noun_1, dim=1)
                head_w1 = torch.cat(head_w1, dim=1)

                all_pred_verb_2 = torch.cat(all_pred_verb_2, dim=1)
                all_pred_noun_2 = torch.cat(all_pred_noun_2, dim=1)
                head_w2 = torch.cat(head_w2, dim=1)
                cnt = 0
                foo_0, bar_0, foo_1, bar_1, foo_2, bar_2 = 0, 0, 0, 0, 0, 0

                for t in range(self.num_segments):
                    if not all_skip[t]:
                        foo_0 += all_pred_verb_0[:, t, :]
                        bar_0 += all_pred_noun_0[:, t, :]

                        foo_1 += all_pred_verb_1[:, t, :]
                        bar_1 += all_pred_noun_1[:, t, :]

                        foo_2 += all_pred_verb_2[:, t, :]
                        bar_2 += all_pred_noun_2[:, t, :]

                        cnt += 1
                output = ((foo_0/cnt, bar_0/cnt), head_w0,
                          (foo_1/cnt, bar_1/cnt), head_w1,
                          (foo_2/cnt, bar_2/cnt), head_w2,)

            extra_output = {
                'skip': all_skip,
                'time': all_time,
                'ssim': all_ssim,
                'r': all_r,
            }

            # Collect batch
            outputs.append(output)
            extra_outputs.append(extra_output)

        # =====================================================================
        # Manipulate batch and output fields
        # =====================================================================
        # Get outputs
        if not has_multihead:
            out_verb = torch.cat([x[0] for x in outputs], dim=0)
            out_noun = torch.cat([x[1] for x in outputs], dim=0)
            outputs = (out_verb, out_noun)
        else:
            out_verb_0 = torch.cat([x[0][0] for x in outputs], dim=0)
            out_noun_0 = torch.cat([x[0][1] for x in outputs], dim=0)
            head_w0 = torch.cat([x[1] for x in outputs], dim=0)

            out_verb_1 = torch.cat([x[2][0] for x in outputs], dim=0)
            out_noun_1 = torch.cat([x[2][1] for x in outputs], dim=0)
            head_w1 = torch.cat([x[3] for x in outputs], dim=0)

            out_verb_2 = torch.cat([x[4][0] for x in outputs], dim=0)
            out_noun_2 = torch.cat([x[4][1] for x in outputs], dim=0)
            head_w2 = torch.cat([x[5] for x in outputs], dim=0)
            outputs = ((out_verb_0, out_noun_0), head_w0,
                       (out_verb_1, out_noun_1), head_w1,
                       (out_verb_2, out_noun_2), head_w2)

        # Combine extra_outputs
        tmp = {}
        tmp['skip'] = np.array([item['skip'] for item in extra_outputs])
        tmp['time'] = np.array([item['time'] for item in extra_outputs])
        tmp['ssim'] = torch.cat([item['ssim'] for item in extra_outputs], dim=0)
        tmp['r'] = torch.cat([item['r'] for item in extra_outputs], dim=0)
        extra_outputs = tmp

        # Compute efficiency loss
        loss_eff, loss_usage, gflops_lst = self.compute_efficiency_loss(extra_outputs['r'])
        # self._check_skip(gflops_lst, extra_outputs['skip'])
        gflops_lst = torch.tensor(gflops_lst).to(loss_eff.device)

        if get_extra:
            return outputs, loss_eff, loss_usage, gflops_lst, extra_outputs
        return outputs, loss_eff, loss_usage, gflops_lst

    def _check_skip(self, gflops_lst, skip):
        full = np.zeros(gflops_lst.shape)
        full[np.where(gflops_lst == self.gflops_full)] = 1

        prescan = np.zeros(gflops_lst.shape)
        prescan[np.where(gflops_lst == self.gflops_prescan)] = 1

        real_skip = np.zeros(gflops_lst.shape)
        real_skip[np.where(gflops_lst == 0)] = 1

        assert np.all(full + prescan + real_skip == 1)
        assert np.all(prescan + real_skip == skip)

    def compute_efficiency_loss(self, r):
        """Compute the efficient loss based on sampling and models FLOPS

        Args:
            r: sampling vector of shape (N, T, max_frames_skip+1) that allows backward

        Return:
            loss_eff: efficiency loss of shape (N,)
        """
        batch_size = r.shape[0]
        loss_eff = torch.zeros([batch_size]).to(self.device)

        gflops_lst = np.zeros([batch_size, self.num_segments])
        for i in range(batch_size):
            gflops_lst[i, 0] = self.gflops_full

            # Ignore the 1st frame because of constant flops (warmup)
            t = 1
            while t < self.num_segments:
                skip = r[i, t].argmax().item()
                if skip == 0:
                    # Full pipeline
                    loss_eff[i] += r[i, t].sum() * self.gflops_full
                    gflops_lst[i, t] = self.gflops_full
                    t += 1
                elif skip == 1:
                    # Run prescan and skip the rest of the pipeline + go to next frame
                    loss_eff[i] += r[i, t].sum() * self.gflops_prescan
                    gflops_lst[i, t] = self.gflops_prescan
                    t += 1
                else:
                    # Run prescan and skip the rest of the pipeline + skip frames
                    loss_eff[i] += r[i, t].sum() * self.gflops_prescan
                    gflops_lst[i, t] = self.gflops_prescan
                    t += skip

        loss_eff *= self.eff_loss_weights
        return loss_eff, torch.zeros_like(loss_eff), gflops_lst

    def warmup(self, low_feat, attn, spec_feat, rgb_high):
        """Warm up to generate memory and avoid skipping the 1st frame. Similar
        to forward_frame(), but without temporal_sampler and using the full
        pipeline.

        Args:
            attn: attention wrt the prescan features
            spec: spectrogram input of time t
            rgb_high: high resolution rgb input of time t

        Return:
            Dictionary of:
                - pred: label prediction of time t
                - hallu: hallucination of attention for time t+1 (from time t)
                - actreg_mem: updated memory of actreg_model
                - hallu_mem: updated memory of hallu_model
                - skipped: whether frame t is skipped
                - ssim: ssim value between attention and hallucination
                - r_t: sampling results from time sampler
                - remaining_skips: new remaining number of frames to skip
        """
        batch_size = rgb_high.shape[0]
        assert batch_size == 1, 'Only support batch_size 1 when forward a frame'

        # Hallucinate ---------------------------------------------------------
        hallu, hallu_mem = self.hallu_model(attn.unsqueeze(dim=1), None)
        assert hallu.shape[1] == 1
        hallu = hallu[:, 0]

        # Spatial sampler -----------------------------------------------------
        # Compute bboxes -> (B, top_k, 4)
        bboxes = self.spatial_sampler.sample_frame(attn, rgb_high.shape[-1],
                                                   reorder_pair=True)

        # Extract regions and feed in high_feat_model
        high_feat = []
        for k in range(self.spatial_sampler.top_k):
            top = bboxes[0, k, 0]
            left = bboxes[0, k, 1]
            bottom = bboxes[0, k, 2]
            right = bboxes[0, k, 3]

            region = rgb_high[:, :, top:bottom, left:right]
            high_feat_k = self.high_feat_model({self._pivot_mod_name: region})
            high_feat.append(high_feat_k)

        # Action recognition --------------------------------------------------
        if self.feat_process_type == 'add':
            all_feats = low_feat + spec_feat
            for k in range(self.spatial_sampler.top_k):
                all_feats += high_feat[k]
        elif self.feat_process_type == 'cat':
            all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=1)

        assert all_feats.ndim == 2
        all_feats = all_feats.unsqueeze(dim=1)

        pred, actreg_mem = self.actreg_model(all_feats, None)

        output = {
            'pred': pred,
            'hallu': hallu,
            'actreg_mem': actreg_mem,
            'hallu_mem': hallu_mem,
            'skipped': False,
            'ssim': torch.zeros([1, 1]).to(self.device),
            'r_t': torch.zeros([1, self.temporal_sampler.max_frames_skip+1]).to(self.device),
            'remaining_skips': 0,
        }
        return output

    def forward_frame(self, low_feat, attn, spec_feat, rgb_high,
                      old_pred=None, old_hallu=None,
                      actreg_mem=None, hallu_mem=None,
                      remaining_skips=0):
        """Forward a single frame

        Args:
            prescan_feat: feature from the first half of the network, using low
                resolution input
            attn: attention wrt the prescan features
            spec: spectrogram input of time t
            rgb_high: high resolution rgb input of time t
            old_pred: label prediction of time t-1
            old_hallu: hallucination of attention for frame t (from time t-1)
            actreg_mem: memory of actreg_model from time t-1
            hallu_mem: memory of hallu_model from time t-1
            remaining_skips: current remaining number of frames to skip

        Return:
            Dictionary of:
                - pred: label prediction of time t
                - hallu: hallucination of attention for time t+1 (from time t)
                - actreg_mem: updated memory of actreg_model
                - hallu_mem: updated memory of hallu_model
                - skipped: whether frame t is skipped
                - ssim: ssim value between attention and hallucination
                - r_t: sampling results from time sampler
                - remaining_skips: new remaining number of frames to skip
        """
        batch_size = rgb_high.shape[0]
        assert batch_size == 1, 'Only support batch_size 1 when forward a frame'

        # =====================================================================
        # Case 0: Still during skipping duration, propagate memory only
        # =====================================================================
        if remaining_skips > 0:
            output = {
                'pred': old_pred,  # Don't update prediction
                'hallu': old_hallu,  # Don't update hallucination
                'actreg_mem': actreg_mem,
                'hallu_mem': hallu_mem,
                'skipped': True,
                'ssim': torch.zeros([1, 1]).to(self.device),
                'r_t': torch.zeros([1, self.temporal_sampler.max_frames_skip+1]).to(self.device),
                'remaining_skips': remaining_skips-1,
            }
            return output

        # =====================================================================
        # Hallucinate and run temporal sampler to decide skipping
        # =====================================================================
        # First half of feature low rgb feat
        # prescan_feat = self.first_half_forward(rgb_low, self.low_feat_model.rgb)

        # Get attention of current frame --> (B, C, H, W)
        # attn = self.low_feat_model.rgb.get_attention_weight(
        #     l_name=self.attention_layer[0],
        #     m_name=self.attention_layer[1],
        #     aggregated=True,
        # )
        # assert attn is not None, 'Fail to retrieve attention'

        # Hallucinate attention of time t+1, using real attention of time t
        hallu, hallu_mem = self.hallu_model(attn.unsqueeze(dim=1), hallu_mem)
        assert hallu.shape[1] == 1
        hallu = hallu[:, 0]

        # Temporal sampler
        r_t, ssim = self.temporal_sampler.sample_frame(
            attn, old_hallu, self.temperature)

        # Update skipping
        for j in range(self.temporal_sampler.max_frames_skip+1):
            if remaining_skips < 0.5 and r_t[0, j] > 0.5:
                remaining_skips = j

        to_skip = remaining_skips > 0

        # =====================================================================
        # Case 1: skipping the current frame without running classification
        # =====================================================================
        if to_skip:
            output = {
                'pred': old_pred,           # Don't update prediction
                'hallu': hallu,             # New hallucination
                'actreg_mem': actreg_mem,   # Don't update actreg_mem
                'hallu_mem': hallu_mem,     # New hallu_mem
                'skipped': True,
                'ssim': ssim,
                'r_t': r_t,
                'remaining_skips': remaining_skips-1,
            }
            return output

        # =====================================================================
        # Case 2: not skipping the current frame
        # =====================================================================
        # Second half of RGB low
        # low_feat = self.second_half_forward(prescan_feat, self.low_feat_model.rgb)

        # Spec
        # spec_feat = self.low_feat_model.spec(spec)

        # Spatial sampler -----------------------------------------------------
        # Compute bboxes -> (B, top_k, 4)
        bboxes = self.spatial_sampler.sample_frame(attn, rgb_high.shape[-1],
                                                   reorder_pair=True)

        # Extract regions and feed in high_feat_model
        high_feat = []
        for k in range(self.spatial_sampler.top_k):
            top = bboxes[0, k, 0]
            left = bboxes[0, k, 1]
            bottom = bboxes[0, k, 2]
            right = bboxes[0, k, 3]

            region = rgb_high[:, :, top:bottom, left:right]
            high_feat_k = self.high_feat_model({self._pivot_mod_name: region})
            high_feat.append(high_feat_k)

        # Action recognition --------------------------------------------------
        if self.feat_process_type == 'add':
            all_feats = low_feat + spec_feat
            for k in range(self.spatial_sampler.top_k):
                all_feats += high_feat[k]
        elif self.feat_process_type == 'cat':
            all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=1)

        assert all_feats.ndim == 2
        all_feats = all_feats.unsqueeze(dim=1)

        pred, actreg_mem = self.actreg_model(all_feats, actreg_mem)

        output = {
            'pred': pred,
            'hallu': hallu,
            'actreg_mem': actreg_mem,
            'hallu_mem': hallu_mem,
            'skipped': False,
            'ssim': ssim,
            'r_t': r_t,
            'remaining_skips': remaining_skips,  # remaining_skips should be 0 now
        }
        assert remaining_skips == 0
        return output

    def first_half_forward(self, x, san):
        """Warper of the forward function of SAN to run the first half
        """
        x = san.relu(san.bn_in(san.conv_in(x)))

        if self.attention_layer == ['layer3', '0']:
            x = san.relu(san.bn0(san.layer0(san.conv0(san.pool(x)))))
            x = san.relu(san.bn1(san.layer1(san.conv1(san.pool(x)))))
            x = san.relu(san.bn2(san.layer2(san.conv2(san.pool(x)))))
            # Run layer3_0
            x = san.layer3[0](san.conv3(san.pool(x)))
        elif self.attention_layer == ['layer2', '3']:
            x = san.relu(san.bn0(san.layer0(san.conv0(san.pool(x)))))
            x = san.relu(san.bn1(san.layer1(san.conv1(san.pool(x)))))
            x = san.relu(san.bn2(san.layer2(san.conv2(san.pool(x)))))
        else:
            raise NotImplementedError
        return x

    def second_half_forward(self, x, san):
        """Warper of the forward function of SAN to run the second half
        """
        if self.attention_layer == ['layer3', '0']:
            # Run from layer3_1 to the end of layer3
            x = san.relu(san.bn3(san.layer3[1:](x)))
            x = san.relu(san.bn4(san.layer4(san.conv4(san.pool(x)))))
        elif self.attention_layer == ['layer2', '3']:
            x = san.relu(san.bn3(san.layer3(san.conv3(san.pool(x)))))
            x = san.relu(san.bn4(san.layer4(san.conv4(san.pool(x)))))
        else:
            raise NotImplementedError

        x = san.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

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
        if self.temporal_sampler_lr is None:
            param_groups = filter(lambda p: p.requires_grad, self.parameters())
        else:
            # Overwrite the lr for temporal_sampler if necessary
            param_groups = []
            param_groups.append({'params': filter(lambda p: p.requires_grad,
                                                  self.temporal_sampler.parameters()),
                                 'lr': self.temporal_sampler_lr})

            for module in [self.low_feat_model, self.high_feat_model, self.spatial_sampler,
                           self.hallu_model, self.actreg_model]:
                try:
                    param_groups.append({'params': filter(lambda p: p.requires_grad,
                                                          module.parameters())})
                except AttributeError:
                    continue
        return param_groups
