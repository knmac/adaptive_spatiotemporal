"""Temporal sampler with threshold
"""
from .pytorch_ssim.ssim import SSIM


class TemporalSamplerThres():
    def __init__(self, threshold, n_frames_to_reset, attention_dim):
        assert -1.0 <= threshold <= 0.0, \
            'Threshold must be in [-1, 0]. Received: {}'.format(threshold)

        self.threshold = threshold
        self.n_frames_to_reset = n_frames_to_reset

        self.frame_cnt = 0
        self.belief_criterion = SSIM(window_size=3, channel=attention_dim[0])

    def reset(self):
        """Reset the frame counter to force running full pipeline after a
        period of time
        """
        self.frame_cnt = 0

    def sample_frame(self, attn, old_hallu):
        """Decide whether to skip a frame by comparing attention and hallucination

        Args:
            attn: attention tensor (N, C, H, W)
            hallu: hallucination tensor (N, C, H, W)

        Return:
            to_skip: whether to skip the frame
            ssim: ssim score of attn and old_hallu
        """
        # Force running full pipeline after a period of time
        if self.frame_cnt >= self.n_frames_to_reset:
            self.reset()
            to_skip = False
            return to_skip, 0.0

        # Initial frame do not have hallucination -> do not skip
        if old_hallu is None:
            to_skip = False
            return to_skip, 0.0

        # Compute ssim score
        ssim = -self.belief_criterion(attn, old_hallu).item()

        # Match attention and hallucination
        # ssim in [-1, 0], lower ssim means more accurate
        if ssim < self.threshold:
            to_skip = True
            self.reset()
        else:
            to_skip = False
            self.frame_cnt += 1
        return to_skip, ssim
