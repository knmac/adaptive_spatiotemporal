import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    consensus_type = 'avg'
    dim = -1
    shape = None

    @staticmethod
    def forward(ctx, input_tensor):
        SegmentConsensus.shape = input_tensor.size()

        if SegmentConsensus.consensus_type == 'avg':
            output = input_tensor.mean(dim=SegmentConsensus.dim, keepdim=True)
        elif SegmentConsensus.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if SegmentConsensus.consensus_type == 'avg':
            _shape = SegmentConsensus.shape
            _dim = SegmentConsensus.dim

            new_shape = list(grad_output.shape)
            new_shape[_dim] = _shape[_dim]

            grad_in = grad_output.expand(new_shape).clone() / float(_shape[_dim])
        elif SegmentConsensus.consensus_type == 'identity':
            grad_in = grad_output.clone()
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        # return SegmentConsensus(self.consensus_type, self.dim)(input)
        SegmentConsensus.consensus_type = self.consensus_type
        SegmentConsensus.dim = self.dim

        return SegmentConsensus.apply(input)
