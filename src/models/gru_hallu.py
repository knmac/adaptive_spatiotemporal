"""GRU with future hallucination

Ref:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    https://github.com/b-etienne/Seq2seq-PyTorch
    https://github.com/IDKiro/action-recognition
"""
import numpy as np
import torch
from torch import nn
from torch.nn.init import normal_, constant_

from .base_model import BaseModel


class GRUHallu(BaseModel):
    """GRU with future hallucination"""

    def __init__(self, device, modality, num_segments, num_class, dropout,
                 feature_dim, rnn_input_size, rnn_hidden_size, rnn_num_layers,
                 hallu_dim):
        """
        Args:
            feature_dim: feature dimension of each modality
            rnn_input_size: input dim of RNN. There is an fc layer to transform
                from (feature_dim*len(modality)) to rnn_input_size
            rnn_hidden_size: dim of hidden features in RNN
            rnn_num_layers: number of hidden layers in RNN
            hallu_dim: dimenion of the hallucinated attention
        """
        super(GRUHallu, self).__init__(device)

        self.modality = modality
        self.num_segments = num_segments
        self.num_class = num_class
        self.dropout = dropout

        self.feature_dim = feature_dim
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.hallu_dim = hallu_dim

        # Prepare some generic layers and variables
        self.relu = nn.ReLU()
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        _std = 0.001

        # Fusion layer
        self.fc1 = nn.Linear(len(modality)*feature_dim, self.rnn_input_size)
        normal_(self.fc1.weight, 0, _std)
        constant_(self.fc1.bias, 0)

        # RNN
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )
        self.rnn.to(self.device)

        # Hallucination layers
        self.fc_hallu = nn.Linear(self.rnn_hidden_size, np.prod(self.hallu_dim))
        normal_(self.fc_hallu.weight, 0, _std)
        constant_(self.fc_hallu.bias, 0)

        # Classification layers
        if isinstance(self.num_class, (list, tuple)):
            self.fc_verb = nn.Linear(self.rnn_hidden_size, self.num_class[0])
            self.fc_noun = nn.Linear(self.rnn_hidden_size, self.num_class[1])
            normal_(self.fc_verb.weight, 0, _std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, _std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(self.rnn_hidden_size, self.num_class)
            normal_(self.fc_action.weight, 0, _std)
            constant_(self.fc_action.bias, 0)

    def forward(self, x, hidden=None):
        """Forward function

        Args:
            x: input tensor of shap (B*T, D)

        Return:
            output: classification results
            hallu: hallucination results
        """
        # Fix the warning: RNN module weights are not part of single contiguous
        # chunk of memory. This means they need to be compacted at every call,
        # possibly greatly increasing memory usage. To compact weights again
        # call flatten_parameters().
        self.rnn.flatten_parameters()

        # Fusion with fc layer
        x = self.relu(self.fc1(x))

        # (B*T, D) --> (B, T, D)
        x = x.view(-1, self.num_segments, self.rnn_input_size)

        # RNN
        x, _ = self.rnn(x, hidden)
        x = self.relu(x)

        # Hallucination with the whole sequence
        hallu = self.hallucinate(x)

        # Classification using the last output of the time sequence
        last_x = x[:, -1, :]
        output = self.classify(last_x)
        return output, hallu

    def hallucinate(self, x):
        """Hallucinate the attention at each time frame

        Args:
            x: input feature of shape (B, T, D)
        """
        B, T, D = x.shape
        assert T == self.num_segments
        assert D == self.rnn_hidden_size

        # (B, T, D) --> (B*T, D)
        x = x.reshape(B*T, D)

        # (B*T, D) --> (B*T, C*H*W)
        x = self.relu(self.fc_hallu(x))

        # (B*T, C*H*W) --> (B, T, C, H, W)
        x = x.reshape([B, T] + self.hallu_dim)
        return x

    def classify(self, x):
        """Classification layer
        """
        # TODO: check if dropout is needed
        if self.dropout > 0:
            x = self.dropout_layer(x)

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            out_verb = self.fc_verb(x)

            # Noun
            out_noun = self.fc_noun(x)

            output = (out_verb, out_noun)
        else:
            x = self.fc_action(x)

        return output

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
