"""Average pooling with future hallucination
"""
import numpy as np
import torch
from torch import nn
from torch.nn.init import normal_, constant_

from .base_model import BaseModel


class AvgHallu(BaseModel):
    """Average pooling with future hallucination"""

    def __init__(self, device, modality, num_segments, num_class, dropout,
                 feature_dim, fusion_dim, hallu_dim):
        """
        Args:
            feature_dim: feature dimension of each modality
            fusion_dim: dimension to fuse the modalities
            hallu_dim: dimenion of the hallucinated attention
        """
        super(AvgHallu, self).__init__(device)

        self.modality = modality
        self.num_segments = num_segments
        self.num_class = num_class
        self.dropout = dropout

        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
        self.hallu_dim = hallu_dim

        # Prepare some generic layers and variables
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        _std = 0.001

        # Fusion layer
        self.fc1 = nn.Linear(len(modality)*feature_dim, self.fusion_dim)
        normal_(self.fc1.weight, 0, _std)
        constant_(self.fc1.bias, 0)

        # Hallucination layers
        self.fc_hallu = nn.Linear(self.fusion_dim, np.prod(self.hallu_dim))
        normal_(self.fc_hallu.weight, 0, _std)
        constant_(self.fc_hallu.bias, 0)

        # Classification layers
        if isinstance(self.num_class, (list, tuple)):
            self.fc_verb = nn.Linear(self.fusion_dim, self.num_class[0])
            self.fc_noun = nn.Linear(self.fusion_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, _std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, _std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(self.fusion_dim, self.num_class)
            normal_(self.fc_action.weight, 0, _std)
            constant_(self.fc_action.bias, 0)

    def forward(self, x):
        """Forward function

        Args:
            x: input tensor of shap (B*T, D)

        Return:
            output: classification results
            hallu: hallucination results
        """
        # Fusion with fc layer
        x = self.relu(self.fc1(x))

        # Hallucination with the whole sequence
        hallu = self.hallucinate(x)

        # Classification with average pooling after softmax
        output = self.classify(x)
        return output, hallu

    def hallucinate(self, x):
        """Hallucinate the attention at each time frame

        Args:
            x: input feature of shape (B*T, D)
        """
        # (B*T, D) --> (B*T, C*H*W)
        x = self.relu(self.fc_hallu(x))

        # (B*T, C*H*W) --> (B, T, C, H, W)
        x = x.reshape([-1, self.num_segments] + self.hallu_dim)
        return x

    def classify(self, x):
        """Classification layer with average pooling for time domain
        """
        # TODO: check if dropout is needed
        if self.dropout > 0:
            x = self.dropout_layer(x)

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            out_verb = self.fc_verb(x)
            # out_verb = self.softmax(out_verb)
            out_verb = out_verb.view((-1, self.num_segments) + out_verb.shape[1:])
            out_verb = torch.mean(out_verb, dim=1)

            # Noun
            out_noun = self.fc_noun(x)
            # out_noun = self.softmax(out_noun)
            out_noun = out_noun.view((-1, self.num_segments) + out_noun.shape[1:])
            out_noun = torch.mean(out_noun, dim=1)

            output = (out_verb, out_noun)
        else:
            output = self.fc_action(x)
            # output = self.softmax(output)
            output = output.view((-1, self.num_segments) + output.shape[1:])
            output = torch.mean(output, dim=1)

        return output
