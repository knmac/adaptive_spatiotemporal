"""Conv LSTM with future hallucination using conv decoder
"""
import torch
from torch import nn
from torch.nn.init import normal_, constant_

from .base_model import BaseModel
from .convlstm import ConvLSTM


class ConvLSTMConvHallu(BaseModel):
    """ConvLSTM with future hallucination"""

    def __init__(self, device, modality, num_segments, num_class, dropout,
                 feature_dim, rnn_input_size, rnn_hidden_size, rnn_num_layers,
                 hallu_dim):
        """
        Args:
            feature_dim: feature dimension of each modality
            rnn_input_size: input dim of RNN. There is a conv layer to transform
                from (feature_dim*len(modality)) to rnn_input_size
            rnn_hidden_size: dim of hidden features in RNN
            rnn_num_layers: number of hidden layers in RNN
            hallu_dim: dimenion of the hallucinated attention
        """
        super(ConvLSTMConvHallu, self).__init__(device)

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
        self.conv_fus = nn.Conv2d(
            in_channels=len(modality)*feature_dim,
            out_channels=self.rnn_input_size,
            kernel_size=3,
            padding=1,
        )

        # RNN
        self.rnn = ConvLSTM(
            input_dim=self.rnn_input_size,
            hidden_dim=self.rnn_hidden_size,
            kernel_size=(3, 3),
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.rnn.to(self.device)

        # Hallucination layers
        self.conv0 = nn.Conv2d(in_channels=self.rnn_hidden_size, out_channels=64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        # Fusion layer
        x = self.relu(self.conv_fus(x))

        # (B*T, C, H, W) --> (B, T, C, H, W)
        x = x.view((-1, self.num_segments) + x.shape[1:])

        # RNN
        x, _ = self.rnn(x, hidden)
        x = x[-1]  # Get the output of the last RNN layer
        x = self.relu(x)

        # Hallucination with the whole sequence
        hallu = self.hallucinate(x)

        # Classification using the last output of the time sequence
        last_x = x[:, -1]
        last_x = self.avgpool(last_x).view(last_x.shape[0], -1)
        output = self.classify(last_x)
        return output, hallu

    def hallucinate(self, x):
        """Hallucinate the attention at each time frame

        Args:
            x: input feature of shape (B, T, D)
        """
        B, T, C, H, W = x.shape
        assert T == self.num_segments
        assert C == self.rnn_hidden_size

        # (B, T, C, H, W) --> (B*T, C, H, W)
        x = x.reshape(B*T, C, H, W)

        x = self.relu(self.bn0(self.conv0(x)))  # (?, 64, 7, 7)
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
