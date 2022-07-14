"""Self-Attention Network

Ref: https://github.com/hszhao/SAN
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sa.modules import Subtraction, Subtraction2, Aggregation
from .sa.modules_noncupy import SubtractionNonCupy, Subtraction2NonCupy, AggregationNonCupy


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1, using_cupy=True):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            if using_cupy:
                self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
                self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            else:
                self.subtraction = SubtractionNonCupy(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
                self.subtraction2 = Subtraction2NonCupy(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        if using_cupy:
            self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        else:
            self.aggregation = AggregationNonCupy(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        self._weight = w  # Store the weights
        return x


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1, using_cupy=True):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride, using_cupy=using_cupy)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out


class SAN(nn.Module):
    def __init__(self, sa_type, block, layers, kernels, num_classes, using_cupy=True):
        super(SAN, self).__init__()
        self.using_cupy = using_cupy
        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

        self._norm_mask_dict = {}  # Store precomputed normalization masks

    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride, using_cupy=self.using_cupy))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        # Use the avgpool layer, if not removed yet
        if hasattr(self, 'avgpool'):
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        # Use the final fc layer, if not removed yet
        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x

    def get_all_attention_weights(self, aggregated, normalize=False):
        """Get attention weights in all SAM modules of the model. Require to
        run forward first.

        Args:
            aggregated: (bool) whether to aggregate the attention weights
            normalize: (bool) whether to normalize the attention weights. Only
                works if aggregated is True

        Return:
            att: an ordered dictionary as follow:
                {`layer_name`: {
                    `module_name`: weights,
                    `module_name`: weights,
                    ...
                 },
                 `layer_name`: {
                    `module_name`: weights,
                    ...
                 },
                 ...  }
        """
        att = OrderedDict()
        for l_name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, l_name)
            att[l_name] = OrderedDict()
            for m_name in layer._modules.keys():
                item = layer._modules[m_name].sam
                if hasattr(item, '_weight'):
                    _weight = item._weight
                    if aggregated:
                        _weight = self._aggreate_attention(
                            _weight, item.kernel_size, item.stride, item.dilation,
                            normalize=normalize)

                    att[l_name][m_name] = _weight
                else:
                    att[l_name][m_name] = None
        return att

    def get_attention_weight(self, l_name, m_name, aggregated, normalize=False):
        """Get attention weight at a certain layer and module. Require to run
        forward first

        Args:
            l_name: (str) name of the layer
            m_name: (str) name of the module
            aggregated: (bool) whether to aggregate the weight across footprint
            normalize: (bool) whether to normalize the attention weights. Only
                works if aggregated is True

        Return:
            weight: attention weight at the layer/module needed
        """
        layer = getattr(self, l_name)
        item = layer._modules[m_name].sam
        if not hasattr(item, '_weight'):
            return None

        weight = item._weight
        if aggregated:
            weight = self._aggreate_attention(weight, item.kernel_size,
                                              item.stride, item.dilation,
                                              normalize=normalize)
        return weight

    def _aggreate_attention(self, weight, kernel_size, stride=1, dilation=1,
                            normalize=False):
        """Aggregate the attention weight across footprint using cum sum

        Args:
            weight: attention weight to aggregate
            kernel_size: size of the footprint kernel
            stride: (only support stride=1 for now)
            dilation: (only support dilation=1 for now)

        Return:
            att: aggregated attention weight
        """
        assert stride == 1 and dilation == 1, \
            'Only support stride and dilation of 1 for now'
        assert weight.shape[2] == kernel_size * kernel_size, \
            'Mismatching kernel_size'
        assert weight.shape[3] == int(weight.shape[3]**0.5)**2, \
            'Input shape is not square'
        batch, weight_channels, _, input_size = weight.shape
        input_size = int(input_size**0.5)
        weight = weight.reshape([batch, weight_channels, kernel_size, kernel_size, input_size, input_size])

        out_size = input_size + kernel_size - 1
        att_pad = torch.zeros([batch, weight_channels, out_size, out_size], dtype=torch.float32).to(weight.device)

        # Aggregate by cum sum
        for i in range(kernel_size):
            for j in range(kernel_size):
                att_pad[:, :, i:i+input_size, j:j+input_size] += weight[:, :, i, j]
        pad = kernel_size // 2

        # Normalize
        if normalize:
            H = att_pad.shape[-1] - pad*2
            if (H, kernel_size) not in self._norm_mask_dict:
                # Compute norm_mask if not precomputed
                norm_mask = F.conv2d(torch.ones(1, 1, H, H),
                                     torch.ones(1, 1, kernel_size, kernel_size),
                                     padding=pad*2)
                self._norm_mask_dict[(H, kernel_size)] = norm_mask
            else:
                norm_mask = self._norm_mask_dict[(H, kernel_size)]
            att_pad = att_pad / norm_mask.to(att_pad.device)

        # Unpad
        att = att_pad[:, :, pad:-pad, pad:-pad]
        return att


def san(sa_type, layers, kernels, num_classes):
    model = SAN(sa_type, Bottleneck, layers, kernels, num_classes)
    return model


if __name__ == '__main__':
    net = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    print(net)
    y = net(torch.randn(4, 3, 224, 224).cuda())
    print(y.size())
