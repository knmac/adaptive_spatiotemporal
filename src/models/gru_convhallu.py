"""GRU with future hallucination using conv decoder
"""
import torch
from torch import nn
from torch.nn.init import normal_, constant_

from .base_model import BaseModel


class GRUConvHallu(BaseModel):
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
        super(GRUConvHallu, self).__init__(device)

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
        self.fc_hallu = nn.Linear(self.rnn_hidden_size, 64*7*7)
        normal_(self.fc_hallu.weight, 0, _std)
        constant_(self.fc_hallu.bias, 0)

        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)

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

        # (B*T, D) --> (B*T, 64x7x7)
        x = self.relu(self.fc_hallu(x))  # (?, 64x7x7)
        x = x.reshape(-1, 64, 7, 7)  # (?, 64, 7, 7)

        x = self.relu(self.bn1(self.conv1(x)))  # (?, 64, 7, 7)
        x = self.upsample(x)  # (?, 64, 14, 14)
        x = self.relu(self.bn2(self.conv2(x)))  # (?, 32, 14, 14)
        x = self.relu(self.bn3(self.conv3(x)))  # (?, 32, 14, 14)
        assert x.shape[1:] == torch.Size(self.hallu_dim)

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
