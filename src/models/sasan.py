""" Stand-Alone Self Attention Networks

Ref:
    https://papers.nips.cc/paper/2019/file/3416a75f4cea9109507cacd8e2f2aefc-Paper.pdf
    https://github.com/leaderj1001/Stand-Alone-Self-Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.attn_raw = None

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        self.attn_raw = out
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.attn_raw = None

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        self.attn_raw = out
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


class Bottleneck(nn.Module):
    """Bottleneck for ResNet backbone"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class SASAN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, stem=False):
        super(SASAN, self).__init__()
        self.in_places = 64

        if stem:
            self.init = nn.Sequential(
                # For ImageNet
                AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(4, 4)
            )
        else:
            self.init = nn.Sequential(
                # For ImageNet
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.num_blocks = num_blocks

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride))
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        if hasattr(self, 'fc'):
            out = self.fc(out)

        # attn_dict = self.get_all_attention()
        return out

    def get_attention(self, layer, aggregated=True, norm=True, unpad=True):
        """Get attention map of an attention layer.
        `norm` and `unpad` do not matter if `aggregated` is False

        Ref:
            https://discuss.pytorch.org/t/how-to-unfold-a-tensor-into-sliding-windows-and-then-fold-it-back/55890/7

        Args:
            layer: attention layer
            aggregated: whether to aggregate across footprints by folding
            norm: whether to normalize to eliminate cumulative effect
            unpad: whether to unpad the attention

        Return:
            attn: retrived attention map
        """
        assert hasattr(layer, 'attn_raw'), 'Layer does not have attention'
        assert layer.attn_raw is not None, 'Attention not computed yet'

        attn = layer.attn_raw
        if not aggregated:
            return attn

        # Prepare the dimension
        kernel_size = layer.kernel_size
        stride = layer.stride
        B, N, C, H, W, K = attn.shape
        padding = 2 * (kernel_size//2)

        # (B, N, C, H, W, K) -> (B*N*C, K, H*W)
        attn = attn.reshape([B*N*C, H*W, K]).permute(0, 2, 1)

        # Fold the attenion to aggregate
        attn = F.fold(attn, (H+padding, W+padding),
                      kernel_size=kernel_size, stride=stride)

        # (B*N*C, 1, H_pad, W_pad) -> (B, N*C, H_pad, W_pad)
        attn = attn.reshape([B, N*C, H+padding, W+padding])

        # Normalize the cummulative effect of sliding
        if norm:
            norm_mask = F.conv2d(torch.ones(1, 1, H, H),
                                 torch.ones(1, 1, kernel_size, kernel_size),
                                 padding=padding).to(attn.device)
            attn = attn / norm_mask

        # Unpad the attention
        if unpad:
            attn = attn[:, :, padding//2:padding//2+H, padding//2:padding//2+W]
        return attn

    def get_all_attention(self, aggregated=True, norm=True, unpad=True):
        """Get attention of all layers

        Return:
            attn_dict: dictionary of attention with the structure:
                attn_dict[l_name][m_name], where l_name is layer name and
                m_name is module name
        """
        attn_dict = {}
        for l in range(len(self.num_blocks)):
            l_name = 'layer{}'.format(l+1)
            attn_dict[l_name] = {}
            layer = getattr(self, l_name)
            for m in range(self.num_blocks[l]):
                attn = self.get_attention(
                    layer[m].conv2[0], aggregated, norm, unpad)
                attn_dict[l_name][str(m)] = attn
        return attn_dict


def SASANResNet26(num_classes=1000, stem=False):
    """Wrapper to build SASAN with ResNet26 backbone"""
    return SASAN(Bottleneck, [1, 2, 4, 1], num_classes=num_classes, stem=stem)


def SASANResNet38(num_classes=1000, stem=False):
    """Wrapper to build SASAN with ResNet38 backbone"""
    return SASAN(Bottleneck, [2, 3, 5, 2], num_classes=num_classes, stem=stem)


def SASANResNet50(num_classes=1000, stem=False):
    """Wrapper to build SASAN with ResNet50 backbone"""
    return SASAN(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem)
