"""Epic kitchen dataset"""
import sys
import os
import pickle
from pathlib import Path

import librosa
import torch
import cv2
import numpy as np
import pandas as pd
import skimage.io as sio
from numpy.random import randint
from PIL import Image
from epic_kitchens.meta import training_labels, test_timestamps
from subprocess import Popen, PIPE

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.base_dataset import BaseDataset
from src.datasets.video_record import VideoRecord
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class EpicKitchenDataset(BaseDataset):
    def __init__(self, mode, list_file, new_length, modality, image_tmpl,
                 visual_path='', audio_path='', fps=29.94,
                 resampling_rate=44000, num_segments=3, transform=None,
                 use_audio_dict=True, to_shuffle=True,
                 depth_path='', depth_tmpl='', depth_cache_tmpl='',
                 semantic_path='', semantic_tmpl='', semantic_cache_tmpl='',
                 full_test_split='', has_motion_compensation=False):
        """Initialize the dataset

        Each sample will be organized as a dictionary as follow
        {
            'modality1': [C * num_segments * new_length[modality1], H, W],
            'modality2': ...
        }
        where C is the number of channels of that modality

        Args:
            mode: (str) train, val, or test mode.
            list_file: (dict) dictionary with keys: train/val/test. Each is the
                path to the pickled filed containing the path for
                training/validation/testing. If is empty string, will load the
                default full training list from EPIC-KITCHENS.
            new_length: (dict) number of frames per segment in a sample
            modality: (list) list of modalities to load.
            image_tmpl: (dict) regular expression template to retrieve image filename.
            depth_tmpl: (dict) regular expression template to retrieve depth filename.
            visual_path: (str) path to the directory containing the frames.
            audio_path: (str) path to the directory containing audio files.
                If use_audio_dict is True then this must be the path to the
                pickled dictionary file.
            depth_path: (str) path to the directory containing the depth.
            fps: frame per seconds
            resampling_rate: (int) resampling rate of audio
            num_segments: (int) num segments per sample (refer to the paper)
            transform: (dict) transformation for each modality, depending on
                the mode.
            use_audio_dict: (bool) whether the audio_path is pointed to a
                dictionary or not.
            to_shuffle: (bool) whether to shuffle non-rgb modalities in training
            full_test_split: (str) only used when list_file['test'] is None.
                Determine which split of epic kitchen test to load.
                Choices: ['seen', 'unseen', 'all']
        """
        super().__init__(mode)
        self.name = 'epic_kitchens'
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        logger.info('Loading audio...')
        if audio_path is not None:
            if not use_audio_dict:
                if not os.path.isdir(audio_path):
                    audio_path = os.path.join(root, audio_path)
                self.audio_path = Path(audio_path)
            else:
                if not os.path.isfile(audio_path):
                    audio_path = os.path.join(root, audio_path)
                with open(audio_path, 'rb') as stream:
                    self.audio_path = pickle.load(stream)
        logger.info('Done loading audio!')

        if not os.path.isdir(visual_path):
            visual_path = os.path.join(root, visual_path)
        self.visual_path = visual_path

        if not os.path.isdir(depth_path):
            depth_path = os.path.join(root, depth_path)
        self.depth_path = depth_path

        if not os.path.isdir(semantic_path):
            semantic_path = os.path.join(root, semantic_path)
        self.semantic_path = semantic_path

        if list_file[mode] is not None:
            # Use the given lists
            if not os.path.isfile(list_file[mode]):
                list_file[mode] = os.path.join(root, list_file[mode])
            self.list_file = pd.read_pickle(list_file[mode])
        else:
            # Use the full list for training and there's no validation
            if mode == 'train':
                self.list_file = training_labels()
            elif mode == 'val':
                self.list_file = None
            elif mode == 'test':
                self.list_file = test_timestamps(full_test_split)

        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.depth_tmpl = depth_tmpl
        self.semantic_tmpl = semantic_tmpl
        self.depth_cache_tmpl = depth_cache_tmpl
        self.semantic_cache_tmpl = semantic_cache_tmpl
        self.transform = transform
        self.resampling_rate = resampling_rate
        self.fps = fps
        self.use_audio_dict = use_audio_dict
        self.to_shuffle = to_shuffle
        self.has_motion_compensation = has_motion_compensation

        if 'RGBDiff' in self.modality:
            self.new_length['RGBDiff'] += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        input = {}
        record = self.video_list[index]
        # print(self.list_file.loc[self.list_file.index[index]])

        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # We implement a Temporal Binding Window (TBW) with size same as the action's length by:
            #   1. Selecting different random indices (timestamps) for each modality within segments
            #      (this is similar to using a TBW with size same as the segment's size)
            #   2. Shuffling randomly the segments of Flow, Audio (RGB is the anchor hence not shuffled)
            #      which binds data across segments, hence making the TBW same in size as the action.
            #   Example of an action with 90 frames across all modalities:
            #    1. Synchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [12, 41, 80], Audio: [12, 41, 80]
            #    2. Asynchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [9, 55, 88], Audio: [20, 33, 67]
            #    3. Asynchronous selection of indices per action:
            #       RGB: [12, 41, 80], Flow: [88, 55, 9], Audio: [67, 20, 33]

            if m != 'RGB' and self.mode == 'train':
                if self.to_shuffle:
                    np.random.shuffle(segment_indices)
            # print(m, segment_indices)

            data = self.get(m, record, segment_indices)
            # If there are problems with the current whole video sequence -> skip
            if data is None:
                return None
            # Otherwise return the sample
            img, label = data

            input[m] = img

        return input, label

    def _parse_list(self):
        """Parse from pandas data frame to list of EpicVideoRecord objects"""
        if self.list_file is not None:
            self.video_list = [EpicVideoRecord(tup) for tup in self.list_file.iterrows()]
        else:
            self.video_list = []

    def _log_specgram(self, audio, window_size=10, step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.resampling_rate / 1e3))
        noverlap = int(round(step_size * self.resampling_rate / 1e3))

        spec = librosa.stft(audio, n_fft=511,
                            window='hann',
                            hop_length=noverlap,
                            win_length=nperseg,
                            pad_mode='constant')

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _extract_sound_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / self.fps
        left_sec = centre_sec - 0.639
        right_sec = centre_sec + 0.639
        audio_fname = record.untrimmed_video_name + '.wav'
        if not self.use_audio_dict:
            samples, sr = librosa.core.load(self.audio_path / audio_fname,
                                            sr=None, mono=True)
        else:
            audio_fname = record.untrimmed_video_name
            samples = self.audio_path[audio_fname]

        duration = samples.shape[0] / float(self.resampling_rate)

        left_sample = int(round(left_sec * self.resampling_rate))
        right_sample = int(round(right_sec * self.resampling_rate))

        if left_sec < 0:
            samples = samples[:int(round(self.resampling_rate * 1.279))]

        elif right_sec > duration:
            samples = samples[-int(round(self.resampling_rate * 1.279)):]
        else:
            samples = samples[left_sample:right_sample]

        return self._log_specgram(samples)

    def _load_data(self, modality, record, idx):
        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame + idx
            return [Image.open(os.path.join(self.visual_path,
                                            record.untrimmed_video_name,
                                            self.image_tmpl[modality].format(idx_untrimmed))
                               ).convert('RGB')]
        elif modality == 'Flow':
            idx_untrimmed = int(np.floor((record.start_frame / 2))) + idx
            # idx_untrimmed = record.start_frame + idx
            x_img = Image.open(os.path.join(self.visual_path,
                                            record.untrimmed_video_name,
                                            self.image_tmpl[modality].format('x', idx_untrimmed)
                                            )).convert('L')
            y_img = Image.open(os.path.join(self.visual_path,
                                            record.untrimmed_video_name,
                                            self.image_tmpl[modality].format('y', idx_untrimmed)
                                            )).convert('L')
            return [x_img, y_img]
        elif modality == 'Spec':
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]

    def _sample_indices(self, record, modality):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        # elif record.num_frames[modality] > self.num_segments:
        #     offsets = np.sort(randint(record.num_frames[modality] - self.new_length[modality] + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record, modality):
        tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets

    def get(self, modality, record, indices):
        """Get sample based on the given modality
        """
        images = list()
        if (modality == 'RGB') and (self.has_motion_compensation):
            pivot_inlier = self.find_pivot_inlier(record, indices)

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)

                # Skip if the data are not loaded
                if seg_imgs is None:
                    return None

                # Motion compensation
                if (modality == 'RGB') and (self.has_motion_compensation):
                    warpped_img = self.compensate_motion(seg_imgs[0], record, p, pivot_inlier)
                    seg_imgs = [warpped_img]

                # Collect results
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1

        # Transform data
        process_data = self.transform[modality](images)
        return process_data, record.label


class EpicVideoRecord(VideoRecord):
    """Store pandas dataframe and return data a class property for simpler
    accessibility. The supported properties are:
        untrimmed_video_name: (str) video id that contains the current segment
        start_frame: (int) starting frame index of the current segment
        end_frame: (int) ending frame index of the current segment
        num_frames: (dict) number of frames for different modalities
        label: (dict) verb and noun class of the current segment
    """
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'Flow': (self.end_frame - self.start_frame) / 2,
                'Spec': self.end_frame - self.start_frame,
                }

    @property
    def label(self):
        if 'verb_class' in self._series.keys().tolist():
            label = {'verb': self._series['verb_class'], 'noun': self._series['noun_class']}
        else:  # Fake label to deal with the test sets (S1/S2) that dont have any labels
            label = -10000
        return label
