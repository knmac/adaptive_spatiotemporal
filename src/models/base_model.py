"""Base model"""
import os

import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, device):
        """Initialize the model

        Args:
            device: the device to place the data in
        """
        super().__init__()

        self.device = device

    def save_model(self, path):
        """Save model state dict to a given path

        Args:
            path: path to save the model
        """
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load_model(self, path):
        """Load model state dict from a given path

        Args:
            path: path to model the model
        """
        assert os.path.isfile(path), '{} not found'.format(path)
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def freeze_fn(self, freeze_mode):
        """Freeze parts of the model"""
        pass

    def compute_loss(self, criterion, output, label):
        """Compute the default loss using the given criterion

        This allows cuztomizable losses for different models
        """
        loss = criterion(output, label)
        return loss

    def add_summaries(self, writer, global_step, **kwargs):
        """Add summaries to tensorboard. Child classes need to override this

        Args:
            writer: tensorboard summary writer
            global_step: global step value to record
        """
        pass

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
