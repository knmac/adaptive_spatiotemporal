"""Base dataset"""
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset to inherit from"""
    def __init__(self, mode):
        self.mode = mode
        self.name = None
        assert mode in ['train', 'val', 'test'], \
            'Unsupported mode: {}'.format(mode)
