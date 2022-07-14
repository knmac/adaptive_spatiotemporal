"""Action recognition using GRU
"""
import os

import torch
from torch import nn
from torch.nn.init import normal_, constant_

from .base_model import BaseModel


class ActregGRU2(BaseModel):

    def __init__(self, device, modality, num_class, dropout, feature_dim,
                 rnn_input_size, rnn_hidden_size, rnn_num_layers,
                 consensus_type, extra_dim=0, pretrained=None):
        super(ActregGRU2, self).__init__(device)

        self.modality = modality
        self.num_class = num_class
        self.dropout = dropout

        self.feature_dim = feature_dim
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.consensus_type = consensus_type

        # Prepare some generic layers and variables
        self.relu = nn.ReLU()
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        _std = 0.001

        # Fusion layer for multiple modalities
        # self.fc1 = nn.Linear(len(modality)*feature_dim, rnn_input_size)
        self._input_dim = feature_dim*len(modality) + extra_dim
        if rnn_input_size is None:
            # No fusion layer. Reuse _input_dim as rnn_input_size
            rnn_input_size = self._input_dim
            self.fc1 = None
        elif isinstance(rnn_input_size, int):
            # Single fusion layer
            self.fc1 = nn.Linear(self._input_dim, rnn_input_size)
            normal_(self.fc1.weight, 0, _std)
            constant_(self.fc1.bias, 0)
        elif isinstance(rnn_input_size, list) or isinstance(rnn_input_size, tuple):
            # Multiple fusion layer
            self.fc1 = []
            prev_dim = self._input_dim
            for dim in rnn_input_size:
                tmp = nn.Linear(prev_dim, dim)
                normal_(tmp.weight, 0, _std)
                constant_(tmp.bias, 0)
                self.fc1.append(tmp)
                self.fc1.append(self.relu)
                prev_dim = dim

            # Remove the last relu because already in forward
            self.fc1 = nn.Sequential(*self.fc1[:-1])

            # Use the last dim as the real rnn input dim
            rnn_input_size = rnn_input_size[-1]
        else:
            raise NotImplementedError

        # RNN
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        ).to(self.device)

        # Classification layers
        if isinstance(num_class, (list, tuple)):
            self.fc_verb = nn.Linear(rnn_hidden_size, num_class[0])
            self.fc_noun = nn.Linear(rnn_hidden_size, num_class[1])
            normal_(self.fc_verb.weight, 0, _std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, _std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(rnn_hidden_size, num_class)
            normal_(self.fc_action.weight, 0, _std)
            constant_(self.fc_action.bias, 0)

        # Load pretrained
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if pretrained is not None:
            if not os.path.isfile(pretrained):
                pretrained = os.path.join(root, pretrained)
            self.load_model(pretrained)

    def forward(self, x, hidden=None):
        """
        Args:
            x: input tensor of shape (B, T, D)
            hidden: hidden memory
        """
        self.rnn.flatten_parameters()

        # Fusion with fc layer
        if self.fc1 is not None:
            x = self.relu(self.fc1(x))

        # RNN
        x, hidden = self.rnn(x, hidden)  # Let GRU handles if hidden is None (zero-init)
        x = self.relu(x)

        # Classification
        if self.dropout > 0:
            x = self.dropout_layer(x)

        if isinstance(self.num_class, (list, tuple)):
            out_verb = self.fc_verb(x)
            out_noun = self.fc_noun(x)
            output = (self.consensus(out_verb), self.consensus(out_noun))
        else:
            output = self.fc_action(x)
            output = self.consensus(output)
        return output, hidden

    def consensus(self, data):
        """Consensus over time domain of data tensor

        Args:
            data: tensor of shape (B, T, C)

        Return:
            consensus data wrt consensus_type
        """
        assert data.ndim == 3

        if self.consensus_type == 'full':
            # Return all frames
            return data
        elif self.consensus_type == 'last':
            # Return the last frame
            return data[:, -1]
        elif self.consensus_type == 'avg':
            # Average across frames
            return torch.mean(data, dim=1)
        else:
            print('consensus_type not supported: {}'.format(self.consensus_type))
            raise NotImplementedError
