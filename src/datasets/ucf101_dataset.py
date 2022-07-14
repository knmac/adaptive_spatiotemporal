"""UCF101 dataset"""
import sys
import os

import torch
import numpy as np
from numpy.random import randint  # type: ignore
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.base_dataset import BaseDataset
from src.datasets.video_record import VideoRecord
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)
LBL_OFFSET = 1  # UCF101 labels are 1-indexed
N_CLASSES = 101


class UCF101Dataset(BaseDataset):
    def __init__(self, mode, list_file, modality=['RGB'], image_tmpl='{:04d}.jpg',
                 visual_path='', num_frames_path='', class_ind_path='',
                 num_segments=3, transform={}, new_length={}):
        super(UCF101Dataset, self).__init__(mode)
        self.name = 'ucf101'

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        self.visual_path = MiscUtils.extend_path(visual_path, root, os.path.isdir)

        num_frames_path = MiscUtils.extend_path(num_frames_path, root, os.path.isfile)
        with open(num_frames_path) as f:
            self.num_frames = {x.split(' ')[0]:int(x.split(' ')[1].strip()) \
                               for x in f}

        class_ind_path = MiscUtils.extend_path(class_ind_path, root, os.path.isfile)
        with open(class_ind_path) as f:
            self.class_ind = {x.split(' ')[1].strip():(int(x.split(' ')[0])-LBL_OFFSET) \
                              for x in f}

        if list_file[mode] is not None:
            self.list_file = MiscUtils.extend_path(list_file[mode], root, os.path.isfile)
        assert modality == ['RGB'], 'Only support RGB for now'
        self.modality = modality
        self.image_tmpl = image_tmpl['RGB']
        self.transform = transform['RGB']
        self.new_length = new_length['RGB']

        self.mode = mode
        self.num_segments = num_segments

        self.video_list = self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]

        segment_indices = []
        if self.mode == 'train':
            segment_indices = self._sample_indices(record)
        elif self.mode == 'val':
            segment_indices = self._get_val_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        img, label = self.get(record, segment_indices)
        return {'RGB': img}, label

    def _parse_list(self):
        with open(self.list_file) as f:
            video_list = [UCF101VideoRecord(x.strip(), self.class_ind, self.num_frames) \
                          for x in f]
        return video_list

    def _load_image(self, directory, idx):
        path = os.path.join(self.visual_path, directory, self.image_tmpl.format(idx))
        return [Image.open(path).convert('RGB')]

    def _sample_indices(self, record):
        average_duration = record.num_frames // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames, size=self.num_segments))
        else:
            offsets = list(range(record.num_frames)) + \
                [record.num_frames - 1] * (self.num_segments - record.num_frames)
            offsets = np.array(offsets)
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments:
            tick = record.num_frames / float(self.num_segments)
            offsets = [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
        else:
            offsets = list(range(record.num_frames)) + \
                [record.num_frames - 1] * (self.num_segments - record.num_frames)
        offsets = np.array(offsets)
        return offsets + 1

    def _get_test_indices(self, record):
        tick = record.num_frames / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            images.extend(self._load_image(record.path, int(seg_ind)))
        process_data = self.transform(images)
        return process_data, record.label


class UCF101VideoRecord(VideoRecord):
    def __init__(self, row, class_ind_dict, num_frames_dict):
        self._data = row.split(' ')[0]
        self._label_meaning = self._data.split('/')[0]
        self._label = torch.tensor(class_ind_dict[self._label_meaning])
        self._num_frames = num_frames_dict[self._data]

    @property
    def path(self):
        return self._data.replace('.avi', '')

    @property
    def label(self):
        return self._label

    @property
    def num_frames(self):
        return self._num_frames
