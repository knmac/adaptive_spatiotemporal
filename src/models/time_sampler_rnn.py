"""Temporal sampler with trainable RNN policy
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseModel
from .pytorch_ssim.ssim import SSIM


class TemporalSamplerRNN(BaseModel):
    def __init__(self, device, attention_dim, max_frames_skip,
                 rnn_input_size, rnn_hidden_size, rnn_num_layers,
                 use_attn, use_hallu, use_ssim):
        super(TemporalSamplerRNN, self).__init__(device)

        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.attention_dim = attention_dim
        self.max_frames_skip = max_frames_skip  # Maximum number of frames that can be skipped
        self.use_attn = use_attn
        self.use_hallu = use_hallu
        self.use_ssim = use_ssim

        assert use_attn or use_hallu or use_ssim, 'Must use at least 1 input'
        input_dim = 0
        if use_attn:
            input_dim += np.prod(attention_dim)
        if use_hallu:
            input_dim += np.prod(attention_dim)
        if use_ssim:
            input_dim += 1
            self.belief_criterion = SSIM(window_size=3, channel=attention_dim[0])

        if rnn_input_size is not None:
            self.fc_fus = nn.Linear(input_dim, rnn_input_size)
        else:
            self.fc_fus = None
            rnn_input_size = input_dim  # Overwrite rnn_input_size
        self.fc_out = nn.Linear(rnn_hidden_size, max_frames_skip+1)  # Include no skipping
        self.softmax = nn.Softmax(dim=1)

        self.old_hidden = None  # old hidden memory

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        ).to(self.device)

    def reset(self):
        """Reset memory
        """
        self.old_hidden = None

    def forward(self, dummy):
        """Dummy function to compute complexity"""
        self.sample_frame(dummy[:, 0], dummy[:, 1], 5.)

    def sample_frame(self, attn, old_hallu, temperature):
        """Decide how many frames to skip
        """
        self.rnn.flatten_parameters()

        # Prepare input
        x = []
        if self.use_attn:
            x.append(attn.flatten(start_dim=1))
        if self.use_hallu:
            if old_hallu is not None:
                x.append(old_hallu.flatten(start_dim=1))
            else:
                x.append(torch.zeros([1, np.prod(self.attention_dim)]).to(self.device))
        if self.use_ssim:
            if old_hallu is not None:
                ssim = -self.belief_criterion(attn, old_hallu).unsqueeze(0).unsqueeze(0)
            else:
                ssim = torch.Tensor([[0.0]]).to(self.device)
            x.append(ssim)
        x = torch.cat(x, dim=1).unsqueeze(dim=1)  # (N, 1, C) -> sequence of 1

        # Feed to RNN
        if self.fc_fus is not None:
            x = self.fc_fus(x)
        x, hidden = self.rnn(x, self.old_hidden)
        self.old_hidden = hidden
        out = self.fc_out(x).squeeze(dim=1)

        # Sampling from RNN output
        p_t = torch.log(self.softmax(out).clamp(min=1e-8))
        r_t = torch.cat([F.gumbel_softmax(p_t[b_i:b_i + 1], tau=temperature, hard=True) for b_i in range(p_t.shape[0])])

        return r_t, ssim

    def sample_multiple_frames(self, attn, hallu_model, temperature):
        """Sample a video with multiple frames

        Return:
            r_list: sampling vector
        """
        self.rnn.flatten_parameters()

        batch_size = attn.shape[0]
        num_segments = attn.shape[1]
        remain_skip_vector = torch.zeros(batch_size, 1)
        old_samp_mem = None  # hidden memory for time sampler
        old_hallu_mem = None  # hidden memory for hallucination
        r_all = []

        # Warming up: run the first frame to get hallucination ----------------
        hallu, hallu_mem = hallu_model(attn[:, 0].unsqueeze(dim=1), old_hallu_mem)
        hallu = hallu[:, 0]
        hallu_dim = list(hallu.shape[1:])

        old_hallu_mem = hallu_mem
        old_hallu = hallu
        old_r_t = torch.zeros([batch_size, self.max_frames_skip+1]).to(attn.device)
        old_r_t[:, 0] = 1  # no skipping on the 1st frame
        r_all.append(old_r_t)

        # Remaining frames ----------------------------------------------------
        for t in range(1, num_segments):
            # Prepare input
            x = []
            if self.use_attn:
                x.append(attn[:, t].flatten(start_dim=1))
            if self.use_hallu:
                x.append(old_hallu.flatten(start_dim=1))
            if self.use_ssim:
                ssim = -self.belief_criterion(
                    attn[:, t], old_hallu, size_average=False).unsqueeze(1)
                x.append(ssim)
            x = torch.cat(x, dim=1).unsqueeze(dim=1)  # (N, 1, C) -> sequence of 1

            # Feed input to RNN time sampler
            out, samp_mem = self.rnn(x, old_samp_mem)
            out = self.fc_out(out).squeeze(dim=1)

            # Hallucinate for future
            hallu, hallu_mem = hallu_model(attn[:, t].unsqueeze(dim=1), old_hallu_mem)
            hallu = hallu[:, 0]

            # Sampling using current frame
            p_t = torch.log(self.softmax(out).clamp(min=1e-8))
            r_t = torch.cat(
                [F.gumbel_softmax(p_t[b_i:b_i + 1], tau=temperature, hard=True)
                 for b_i in range(p_t.shape[0])])

            # Update states by batch
            if old_samp_mem is not None:
                take_bool = remain_skip_vector > 0.5
                # take_old = torch.tensor(take_bool, dtype=torch.float).to(attn.device)
                # take_curr = torch.tensor(~take_bool, dtype=torch.float).to(attn.device)
                take_old = take_bool.to(attn.device, torch.float)
                take_curr = 1.0 - take_old

                # GRU memory --> (layer, batch, dim)
                samp_mem = (old_samp_mem * take_old.unsqueeze(0)) + (samp_mem * take_curr.unsqueeze(0))
                r_t = (old_r_t * take_old) + (r_t * take_curr)

                take_old_r = take_old.unsqueeze(-1).unsqueeze(-1).repeat([1] + hallu_dim)
                take_curr_r = take_curr.unsqueeze(-1).unsqueeze(-1).repeat([1] + hallu_dim)
                for ll in range(len(hallu_mem[0])):
                    hallu_mem[0][ll] = (old_hallu_mem[0][ll] * take_old_r) + \
                        (hallu_mem[0][ll] * take_curr_r)
                hallu = (old_hallu * take_old_r) + (hallu * take_curr_r)

            # Update skipping vector
            for batch_i in range(batch_size):
                for skip_i in range(self.max_frames_skip+1):
                    if remain_skip_vector[batch_i][0] < 0.5 and r_t[batch_i][skip_i] > 0.5:
                        remain_skip_vector[batch_i][0] = skip_i

            old_samp_mem = samp_mem
            old_r_t = r_t
            old_hallu_mem = hallu_mem
            old_hallu = hallu
            r_all.append(r_t)
            remain_skip_vector = (remain_skip_vector - 1).clamp(0)

        r_all = torch.stack(r_all, dim=1)
        return r_all
