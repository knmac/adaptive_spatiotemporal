"""Pipeline version 3 - Run only hallucinator

Include teacher forcing ratio
"""
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from .base_model import BaseModel
from .pytorch_ssim.ssim import SSIM
from src.utils.load_cfg import ConfigLoader


class Pipeline3(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim,
                 rnn_prefix_len, tf_decay, feat_model_cfg, hallu_model_cfg,
                 norm_attention=False, using_cupy=True):
        super(Pipeline3, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.norm_attention = norm_attention  # whether to normalize attention
        self.using_cupy = using_cupy

        self.rnn_prefix_len = rnn_prefix_len  # N frames to feed in RNN at a time
        self.init_tf_ratio = 1.0  # Initial teacher forcing ratio
        self.tf_ratio = 1.0
        self.tf_decay = tf_decay

        # Generate feature extraction model
        name, params = ConfigLoader.load_model_cfg(feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
            'using_cupy': self.using_cupy,
        })
        self.feat_model = model_factory.generate(name, device=device, **params)

        # Pivot modality to extract attention
        if 'RGB' in self.modality:
            self._pivot_mod = self.feat_model.rgb
        else:
            raise NotImplementedError

        # Generate hallucination model
        name, params = ConfigLoader.load_model_cfg(hallu_model_cfg)
        assert name in ['HalluConvLSTM2'], \
            'Unsupported model: {}'.format(name)
        params.update({
            'attention_dim': self.attention_dim,
        })
        self.hallu_model = model_factory.generate(name, device=device, **params)

        # Loss for belief propagation
        self.belief_criterion = SSIM(window_size=3, channel=self.attention_dim[0])

    def forward(self, x):
        # Extract features ----------------------------------------------------
        self.feat_model(x)  # Only feed forward to get the attention

        # Retrieve attention from feature model
        attn = self._pivot_mod.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
            normalize=self.norm_attention,
        )
        attn = attn.view([-1, self.num_segments] + list(attn.shape[1:]))
        self._attn = attn

        # Hallucination -------------------------------------------------------
        # Attention shape: [B, T, C, H, W]
        if self.training:
            hallu = self._forward_hallu_train(attn)
        else:
            hallu = self._forward_hallu_val(attn)
        self._hallu = hallu

        # Dummy classification output -----------------------------------------
        # There will be no gradient for the action recognition part. This is
        # only here to make train_val.py happy
        batch_size = attn.shape[0]
        if isinstance(self.num_class, list):
            output = (
                torch.zeros([batch_size, self.num_class[0]]).to(self.device),
                torch.zeros([batch_size, self.num_class[1]]).to(self.device),
            )
        else:
            output = torch.zeros([batch_size, self.num_class]).to(self.device)

        return output, self.compare_belief().unsqueeze(dim=0)

    def _forward_hallu_train(self, inputs):
        """Forward routine of hallucination model for training phase
        Use teacher forcing to improve performance

        Args:
            inputs: attention of shape (B, T, C, H, W)

        Return:
            hallu: hallucination of shape (B, T-rnn_prefix_len, C, H, W)
        """
        target_length = self.num_segments - self.rnn_prefix_len
        predictions = []

        # First batch
        predicted, hidden = self.hallu_model(
            inputs[:, :self.rnn_prefix_len], hidden=None)
        predicted = predicted[:, -1].unsqueeze(dim=1)  # Get only future prediction
        predictions.append(predicted)

        # Remaining batches
        tf_mask = np.random.uniform(size=target_length-1) < self.tf_ratio
        i = 0
        while i < target_length - 1:
            contiguous_frames = 1
            # Batch together consecutive teacher forcing to improve performance
            if tf_mask[i]:
                while (i+contiguous_frames < target_length-1) \
                        and (tf_mask[i+contiguous_frames]):
                    contiguous_frames += 1

                # Feed ground truth
                ix_start = self.rnn_prefix_len + i
                ix_stop = self.rnn_prefix_len + i + contiguous_frames
                predicted, hidden = self.hallu_model(inputs[:, ix_start:ix_stop], hidden)
            else:
                # Feed own output
                predicted, hidden = self.hallu_model(predicted, hidden)
                predicted = predicted[:, -1].unsqueeze(dim=1)  # Get only future prediction

            predictions.append(predicted)
            if contiguous_frames > 1:
                predicted = predicted[:, -1:]
            i += contiguous_frames

        hallu = torch.cat(predictions, dim=1)
        assert hallu.shape[1] == target_length
        return hallu

    def _forward_hallu_val(self, inputs):
        """Forward routine of hallucination model for evaluation phase

        Args:
            inputs: attention of shape (B, T, C, H, W)

        Return:
            hallu: hallucination of shape (B, T-rnn_prefix_len, C, H, W)
        """
        target_length = self.num_segments - self.rnn_prefix_len
        predictions = []

        # First batch
        predicted, hidden = self.hallu_model(
            inputs[:, :self.rnn_prefix_len], hidden=None)
        predicted = predicted[:, -1].unsqueeze(dim=1)  # Get only future prediction
        predictions.append(predicted)

        # Remaining batches
        for i in range(target_length - 1):
            # Feed own output
            predicted, hidden = self.hallu_model(predicted, hidden)
            predicted = predicted[:, -1].unsqueeze(dim=1)  # Get only future prediction
            predictions.append(predicted)

        hallu = torch.cat(predictions, dim=1)
        assert hallu.shape[1] == target_length
        return hallu

    def decay_teacher_forcing_ratio(self, epoch):
        """Decay the teacher forcing ratio `self.tf_ratio` by the factor of
        `self.tf_decay`
        """
        # self.tf_ratio *= self.tf_decay
        self.tf_ratio = self.init_tf_ratio * (self.tf_decay**epoch)

    def compare_belief(self):
        """Compare between attention and hallucination. Do NOT call directly.

        If using multiple GPUs, self._hallu and self._attn will not be available
        """
        assert hasattr(self, '_attn') and hasattr(self, '_hallu'), \
            'Attributes are not found'
        assert torch.all(self._attn >= 0) and torch.all(self._hallu >= 0)

        # Get attention of future frames
        attn_future = self._attn[:, self.rnn_prefix_len:]

        # Get hallucination from current frames (for future frames)
        hallu_current = self._hallu

        # Compare belief
        # Reshape (B,T,C,H,W) --> (B*T,C,H,W) to compare individual images
        # Reverse the sign to maximize SSIM loss
        loss_belief = -self.belief_criterion(
            hallu_current.reshape([-1] + self.attention_dim),
            attn_future.reshape([-1] + self.attention_dim),
        )
        return loss_belief

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
