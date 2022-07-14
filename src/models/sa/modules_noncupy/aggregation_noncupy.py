from torch import nn


class AggregationNonCupy(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(AggregationNonCupy, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, input, weight):
        # return F.aggregation(input, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
        assert (input.shape[0] == weight.shape[0]) and (input.shape[1] % weight.shape[1] == 0) and (self.pad_mode in [0, 1])
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        n, c_x, in_height, in_width = input.shape
        _, c_w, _, _ = weight.shape
        out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
        out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)

        if self.pad_mode == 0:
            # Zero-pad
            unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
            x2 = unfold_j(input).view(n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width)
            out = (weight.unsqueeze(1) * x2).sum(-2).view(n, c_x, out_height, out_width)
        elif self.pad_mode == 1:
            # Ref-pad
            unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            if padding < in_height:
                pad = nn.ReflectionPad2d(padding)
                x2 = unfold_j(pad(input)).view(n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width)
            else:
                pad = nn.ReflectionPad2d(1)
                input_pad = input
                for _ in range(padding):
                    input_pad = pad(input_pad)
                x2 = unfold_j(input_pad).view(n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width)
            out = (weight.unsqueeze(1) * x2).sum(-2).view(n, c_x, out_height, out_width)
        return out
