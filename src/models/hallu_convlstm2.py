"""Hallucination using conv lstm that also return hidden memory
"""
from torch import nn

from .base_model import BaseModel
from .convlstm import ConvLSTM


class HalluConvLSTM2(BaseModel):

    def __init__(self, device, attention_dim,
                 rnn_input_dim, rnn_hidden_dim, rnn_num_layers):
        """
        Args:
            attention_dim: dimenion of the attention
            rnn_input_size: input dim of RNN. There is a conv layer to transform
                from (feature_dim*len(modality)) to rnn_input_size
            rnn_hidden_size: dim of hidden features in RNN
            rnn_num_layers: number of hidden layers in RNN
        """
        super(HalluConvLSTM2, self).__init__(device)

        self.attention_dim = attention_dim
        assert len(attention_dim) == 3, 'attention_dim must be [C, H, W]'

        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        # Prepare some generic layers and variables
        self.relu = nn.ReLU()

        # RNN
        self.rnn = ConvLSTM(
            input_dim=self.rnn_input_dim,
            hidden_dim=self.rnn_hidden_dim,
            kernel_size=(3, 3),
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        ).to(self.device)

        # Encoder
        self.conv_encoder = nn.Conv2d(
            in_channels=attention_dim[0],
            out_channels=rnn_input_dim,
            kernel_size=3,
            padding=1,
        )
        self.bn_encoder = nn.BatchNorm2d(rnn_input_dim)

        # Decoder
        self.conv_decoder = nn.Conv2d(
            in_channels=rnn_hidden_dim,
            out_channels=attention_dim[0],
            kernel_size=3,
            padding=1,
        )
        self.bn_decoder = nn.BatchNorm2d(attention_dim[0])

    def forward(self, x, hidden=None):
        """Forward function

        Args:
            x: input tensor of shap (B, T, C, H, W)
            hidden: input hidden memory

        Return:
            output: hallucination results (B, T, C, H, W)
            hidden: output hidden memory
        """
        if hidden is None:
            hidden = self.rnn._init_hidden(batch_size=x.shape[0],
                                           image_size=(x.shape[-2], x.shape[-1]))

        # ---------------------------------------------------------------------
        # Encoder
        # (B, T, C, H, W) --> (B*T, C, H, W)
        _, T, C, H, W = x.shape
        x = x.reshape((-1, C, H, W))

        # Encode
        x = self.relu(self.bn_encoder(self.conv_encoder(x)))

        # (B*T, C, H, W) --> (B, T, C, H, W)
        _, C, H, W = x.shape
        x = x.reshape((-1, T, C, H, W))

        # ---------------------------------------------------------------------
        # RNN
        x, hidden = self.rnn(x, hidden)
        x = x[-1]  # Get the output of the last RNN layer (all frames)
        x = self.relu(x)

        # ---------------------------------------------------------------------
        # Decoder
        # (B, T, C, H, W) --> (B*T, C, H, W)
        _, T, C, H, W = x.shape
        x = x.reshape((-1, C, H, W))

        # Decode
        x = self.relu(self.bn_decoder(self.conv_decoder(x)))

        # (B*T, C, H, W) --> (B, T, C, H, W)
        _, C, H, W = x.shape
        x = x.reshape((-1, T, C, H, W))

        return x, hidden
