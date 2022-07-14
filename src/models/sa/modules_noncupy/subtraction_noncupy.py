from torch import nn


class SubtractionNonCupy(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(SubtractionNonCupy, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, input):
        # return F.subtraction(input, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
        assert (input.dim() == 4) and (self.pad_mode in [0, 1])
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        n, c, in_height, in_width = input.shape
        out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
        out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)

        if self.pad_mode == 0:
            # Zero-pad
            unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
            out = unfold_i(input).view(n, c, 1, out_height * out_width) - unfold_j(input).view(n, c, pow(kernel_size, 2), out_height * out_width)
        elif self.pad_mode == 1:
            # Ref-pad
            unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            if padding < in_height:
                pad = nn.ReflectionPad2d(padding)
                out = unfold_i(input).view(n, c, 1, out_height * out_width) - unfold_j(pad(input)).view(n, c, pow(kernel_size, 2), out_height * out_width)
            else:
                pad = nn.ReflectionPad2d(1)
                input_pad = input
                for _ in range(padding):
                    input_pad = pad(input_pad)
                out = unfold_i(input).view(n, c, 1, out_height * out_width) - unfold_j(input_pad).view(n, c, pow(kernel_size, 2), out_height * out_width)
        return out
