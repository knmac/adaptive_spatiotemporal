import torch

from .tbn import TBN


class TBNFeat(TBN):
    """Wrapper of TBN to extract feature instead of feeding to
    Fusion_Classification_Network
    """
    def forward(self, input):
        concatenated = []
        # Get the output for each modality
        for m in self.modality:
            if (m == 'RGB'):
                channel = 3
            elif (m == 'Flow'):
                channel = 2
            elif (m == 'Spec'):
                channel = 1
            sample_len = channel * self.new_length[m]

            if m == 'RGBDiff':
                sample_len = 3 * self.new_length[m]
                input[m] = self._get_diff(input[m])
            base_model = getattr(self, m.lower())
            base_out = base_model(input[m].view((-1, sample_len) + input[m].size()[-2:]))

            base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)

        out_feat = torch.cat(concatenated, dim=1)
        return out_feat
