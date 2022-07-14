"""Misc helper methods"""
import io
import os
from random import sample
import sys
import glob
import shutil
import argparse

import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import default_collate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import seaborn as sns

matplotlib.use('Agg')
sns.set_style("whitegrid", {'axes.grid': False})

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.transforms import (
    GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupScale,
    GroupCenterCrop, GroupNormalize, IdentityTransform,
    Stack, ToTorchFormatTensor
)
from tools.complexity import is_supported_instance, flops_to_string, get_model_parameters_number
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class MiscUtils:

    @staticmethod
    def save_progress(model, optimizer, logdir, best_val, epoch, is_best=False):
        """Save the training progress for model and optimizer

        Data are saved as: [logdir]/epoch_[epoch].[extension]

        Args:
            model: model to save
            optimizer: optimizer to save
            logdir: where to save data
            epoch: the current epoch
            is_best: if True, will backup the best model
        """
        prefix = os.path.join(logdir, 'epoch_{:05d}'.format(epoch))
        logger.info('Saving to: %s' % prefix)

        try:
            model.save_model(prefix+'.model')
        except AttributeError:
            model.module.save_model(prefix+'.model')
        # torch.save(optimizer.state_dict(), prefix+'.opt')
        # torch.save(torch.get_rng_state(), prefix+'.rng')
        # torch.save(torch.cuda.get_rng_state(), prefix+'.curng')

        data = {
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'best_prec1': best_val,
        }
        torch.save(data, prefix+'.stat')

        if is_best:
            shutil.copyfile(os.path.join(prefix+'.model'),
                            os.path.join(logdir, 'best.model'))

    @staticmethod
    def load_progress(model, optimizer, device, prefix):
        """Load the training progress for model and optimizer

        Data are loaded from: [prefix].[extension]

        Args:
            model: model to load
            optimizer: optimizer to load
            deivce: device to transfer the optimizer states to
            prefix: prefix with the format [logdir]/epoch_[epoch]

        Return:
            lr: loaded learning rate
            next_epoch: id of the nex epoch
        """
        logger.info('Loading from: %s' % prefix)

        model.load_model(prefix+'.model')
        # optimizer.load_state_dict(torch.load(prefix+'.opt'))
        # torch.set_rng_state(torch.load(prefix+'.rng'))
        # torch.cuda.set_rng_state(torch.load(prefix+'.curng'))

        data = torch.load(prefix+'.stat')
        optimizer.load_state_dict(data['optimizer'])
        torch.set_rng_state(data['rng_state'])
        torch.cuda.set_rng_state(data['cuda_rng_state'])
        best_val = data['best_prec1']

        lr = optimizer.param_groups[0]['lr']
        tmp = os.path.basename(prefix)
        next_epoch = int(tmp.replace('epoch_', '')) + 1

        # Individually transfer the optimizer parts
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return lr, next_epoch, best_val

    @staticmethod
    def get_lastest_checkpoint(logdir, regex='epoch_*.model'):
        """Get the latest checkpoint in a logdir

        For example, for the logdir with:
            logdir/
                epoch_00000.model
                epoch_00001.model
                ...
                epoch_00099.model
        The function will return `logdir/epoch_00099`

        Args:
            logdir: log directory to find the latest checkpoint
            regex: regular expression to describe the checkpoint

        Return:
            prefix: prefix with the format [logdir]/epoch_[epoch]
        """
        assert os.path.isdir(logdir), 'Not a directory: {}'.format(logdir)

        save_lst = glob.glob(os.path.join(logdir, regex))
        save_lst.sort()
        prefix = save_lst[-1].replace('.model', '')
        return prefix

    @staticmethod
    def str2bool(v):
        """Convert a string to boolean type for argparse"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def get_train_augmentation(modality, input_size):
        """Get data augmentation for training phase.
        Copied from epic fusion model's train augmentation.

        Args:
            modality: (dict) dictionary of modality to add transformation for
                augmentation. Different modality requires different transform
            input_size: (dict) dictionary of input size for each modality. We
                assume that the width and height are the same. For example,
                {'RGB': 224, 'Flow': 224, 'Spec': 224}

        Return:
            augmentation: (dict) dictionary of composed transforms for each
                modality
        """
        augmentation = {}
        if 'RGB' in modality:
            augmentation['RGB'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(input_size['RGB'], [1, .875, .75, .66]),
                 GroupRandomHorizontalFlip(is_flow=False)])
        if 'Flow' in modality:
            augmentation['Flow'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(input_size['Flow'], [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=True)])
        if 'RGBDiff' in modality:
            augmentation['RGBDiff'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(input_size['RGBDiff'], [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=False)])
        return augmentation

    @staticmethod
    def get_train_val_transforms(modality, input_mean, input_std, scale_size,
                                 crop_size, train_augmentation,
                                 flow_prefix='', arch='BNInception'):
        """Get transform for train and val phases.
        Copied from epic fusion train.py

        Args:
            modality: (dict) dictionary of modality to add transformation for
                augmentation. Different modality requires different transform
            input_mean: (dict) mean per channels for different modalities, only
                for 'RGB' and 'Flow' modalities, e.g.
                {RGB': [104, 117, 128], 'Flow': [128]}
            input_std: (dict) standard deviation for different modalities, e.g.,
                {'RGB': [1], 'Flow': [1], 'Spec': [1]}
            scale_size: (dict) size to rescale the input to before cropping, e.g.
                {'RGB': 256, 'Flow': 256, 'Spec': 256}
            train_augmentation: (dict) extra augmentation for training phase.
                Refer to get_train_augmentation() here for more information
            flow_prefix: (str) prefix of flow filenames for loading
            arch: (str) name of the architecture. Add special transformations if
                arch == BNInception

        Return:
            train_transform, val_transform: (dict, dict) transformations wrt
                different modalities for training and validation
        """
        normalize = {}
        for m in modality:
            if (m != 'Spec'):
                if (m != 'RGBDiff'):
                    normalize[m] = GroupNormalize(input_mean[m], input_std[m])
                else:
                    normalize[m] = IdentityTransform()

        image_tmpl = {}
        train_transform = {}
        val_transform = {}
        for m in modality:
            if (m in ['RGB', 'Flow', 'RGBDiff']):
                # Prepare dictionaries containing image name templates for each modality
                if m in ['RGB', 'RGBDiff']:
                    image_tmpl[m] = "img_{:010d}.jpg"
                elif m == 'Flow':
                    image_tmpl[m] = flow_prefix + "{}_{:010d}.jpg"
                # Prepare train/val dictionaries containing the transformations
                # (augmentation+normalization)
                # for each modality
                train_transform[m] = torchvision.transforms.Compose([
                    train_augmentation[m],
                    Stack(roll=(arch == 'BNInception')),
                    ToTorchFormatTensor(div=(arch != 'BNInception')),
                    normalize[m],
                ])

                val_transform[m] = torchvision.transforms.Compose([
                    GroupScale(int(scale_size[m])),
                    GroupCenterCrop(crop_size[m]),
                    Stack(roll=(arch == 'BNInception')),
                    ToTorchFormatTensor(div=(arch != 'BNInception')),
                    normalize[m],
                ])
            elif (m == 'Spec'):
                # Prepare train/val dictionaries containing the transformations
                # (augmentation+normalization)
                # for each modality
                train_transform[m] = torchvision.transforms.Compose([
                    Stack(roll=(arch == 'BNInception')),
                    ToTorchFormatTensor(div=False),
                ])

                val_transform[m] = torchvision.transforms.Compose([
                    Stack(roll=(arch == 'BNInception')),
                    ToTorchFormatTensor(div=False),
                ])
        return train_transform, val_transform

    @staticmethod
    def deprocess_rgb(rgb, num_segments, bgr2rgb=True, mean=[104, 117, 128], std=1):
        """Deprocess RGB tensor (multiple frames) for visualization

        Args:
            rgb: tensor of shape ([T*3, H, W])
            num_segments: T in rgb
            bgr2rgb: whether convert from BGR to RGB
            mean, st: the mean and std which was removed from image normalization

        Return:
            Deprocess rgb image
        """
        rgb = rgb.cpu().numpy()
        _, h, w = rgb.shape
        rgb = rgb.reshape([num_segments, 3, h, w]).transpose(0, 2, 3, 1)
        rgb *= std  # std
        rgb += np.array(mean)  # mean
        if bgr2rgb:
            rgb = rgb[..., ::-1]
        rgb = rgb.astype(np.uint8)
        return rgb

    @staticmethod
    def compare_dicts(dict1, dict2, epsilon=1e-7, verbose=False):
        """Compare the content of two dictionaries of torch tensors

        Args:
            dict1: the first dictionary
            dict2: the second dictionary
            epsilon: tolerance factor. If the difference between values of 2
                dictionaries wrt each key is less than epsilon, they are
                considered to be the same. For strict comparison, set epsilon=0
            verbose: whether to print out the difference statistics of a key,
                including mean, standard deviation, and max

        Return:
            all_same: True if all values are the same (wrt to epsilon).
                False otherwise.
        """
        if dict1.keys() != dict2.keys():
            if verbose:
                print('Dictionaries have different keys. Not comparable!')
            return False

        all_same = True
        for k in dict1:
            diff = (dict1[k] - dict2[k]).abs()
            same = diff.max().item() <= epsilon

            if same is False:
                all_same = False
                if verbose:
                    print('>>> %s -> diff: mean=%e, std=%e, max=%e' %
                          (k, diff.mean(), diff.std(), diff.max()))

        if verbose:
            if all_same:
                print('Dictionaries are the same')
            else:
                print('Dictionaries are different')
        return all_same

    @staticmethod
    def ref_bilateral_filter(img_in, img_ref, k_size, sigma_i, sigma_s,
                             reg_constant=1e-8):
        """Bilateral filtering using referencing image. If padding is desired,
        img_in should be padded prior to calling

        Ref: http://jamesgregson.ca/bilateral-filtering-in-python.html

        Args:
            img_in: (ndarray) input image of shape (H, W)
            img_ref: (ndarray) referencing image of shape (H, W)
            k_size: (int) kernel size
            sigma_i: (float) value gaussian std. dev.
            sigma_s: (float) spatial gaussian std. dev.
            reg_constant: (float) optional regularization constant for pathalogical cases

        Returns:
            result: (ndarray) output bilateral-filtered image
        """
        # check the input
        # if not isinstance(img_in, np.ndarray) or img_in.dtype != 'float32' or img_in.ndim != 2:
        #     raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

        # Gaussian function
        def gaussian(x2, sigma2):
            return (1.0 / (2*np.pi*sigma2)) * np.exp(-x2 / (2*sigma2))

        # Half of kernel size
        half_size = k_size//2

        # Initialize results
        wgt_sum = np.ones(img_in.shape)*reg_constant
        result = img_in*reg_constant

        # Bilateral filtering
        for shft_x in range(-half_size, half_size+1):
            for shft_y in range(-half_size, half_size+1):
                # Compute the spatial weight
                gs = gaussian(shft_x**2+shft_y**2, sigma_s**2)

                # Shift by the offsets
                off_in = np.roll(img_in, [shft_y, shft_x], axis=[0, 1])
                off_ref = np.roll(img_ref, [shft_y, shft_x], axis=[0, 1])

                # Compute the value weight
                gi = gaussian((off_ref-img_ref)**2, sigma_i**2)

                # Accumulate the results
                tw = gs * gi
                result += off_in*tw
                wgt_sum += tw
        result /= wgt_sum
        return result

    @staticmethod
    def collect_flops(model, units='GMac', precision=3):
        """Wrapper to collect flops and number of parameters at each layer"""
        total_flops = model.compute_average_flops_cost()

        def accumulate_flops(self):
            if is_supported_instance(self):
                return self.__flops__ / model.__batch_counter__
            else:
                sum = 0
                for m in self.children():
                    sum += m.accumulate_flops()
                return sum

        def flops_repr(self):
            accumulated_flops_cost = self.accumulate_flops()
            return ', '.join([flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                              '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                              self.original_extra_repr()])

        def add_extra_repr(m):
            m.accumulate_flops = accumulate_flops.__get__(m)
            flops_extra_repr = flops_repr.__get__(m)
            if m.extra_repr != flops_extra_repr:
                m.original_extra_repr = m.extra_repr
                m.extra_repr = flops_extra_repr
                assert m.extra_repr != m.original_extra_repr

        def del_extra_repr(m):
            if hasattr(m, 'original_extra_repr'):
                m.extra_repr = m.original_extra_repr
                del m.original_extra_repr
            if hasattr(m, 'accumulate_flops'):
                del m.accumulate_flops

        model.apply(add_extra_repr)
        # print(model, file=ost)

        # Retrieve flops and param at each layer and sub layer (2 levels)
        flops_dict, param_dict = {}, {}
        for i in model._modules.keys():
            item = model._modules[i]
            if isinstance(model._modules[i], torch.nn.modules.container.Sequential):
                for j in model._modules[i]._modules.keys():
                    key = '{}-{}'.format(i, j)
                    flops_dict[key] = item._modules[j].accumulate_flops()
                    param_dict[key] = get_model_parameters_number(item._modules[j])
            else:
                flops_dict[i] = item.accumulate_flops()
                param_dict[i] = get_model_parameters_number(item)

        model.apply(del_extra_repr)
        return flops_dict, param_dict

    @staticmethod
    def safe_collate(batch):
        """Safe collate for data loader to skip broken samples (returned as None)
        """
        batch = list(filter(lambda x: x is not None, batch))
        # for python2: batch = filter(lambda x: x is not None, batch)
        if len(batch) == 0:
            batch = [(-1, -1)]
        return default_collate(batch)

    @staticmethod
    def find_points_correspondence(ptid_1, ptid_2, pts_1, pts_2):
        """Find the point correspondence by looking at point id

        Args:
            ptid_1: (list) point id of the 1st frame
            ptid_2: (list) point id of the 2nd frame
            pts_1: (dict) point data of the 1st frame. The key has to match ptid_1
            pts_2: (dict) point data of the 2nd frame. The key has to match ptid_2

        Return:
            matched_1: (ndarray) matched points from the 1st frame
            matched_2: (ndarray) matched points from the 2nd frame
        """
        common_ids = np.intersect1d(ptid_1, ptid_2)
        matched_1 = np.array([pts_1[k] for k in common_ids], dtype=np.float32)
        matched_2 = np.array([pts_2[k] for k in common_ids], dtype=np.float32)
        return matched_1, matched_2

    @staticmethod
    def fig2img(fig):
        """Convert from matplotlib figure to PIL image

        Args:
            fig: figure handler from matplotlib

        Return:
            fig_img: figure image as Tensor
        """
        # Get buffered image
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        fig_img = Image.open(buf)
        fig_img = ToTensor()(fig_img)
        return fig_img

    @staticmethod
    def plot_grad_flow(named_parameters, show=True):
        """Plot the gradients of all layers with requires_grad

        Args:
            named_parameters: model's layers
            show: if `True` will show a figure of the gradient flow, otherwise
                will return an image tensor

        Return:
            Image tensor if show is False
        """
        matplotlib.use('Agg')
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

        fig = plt.figure(figsize=(12, 10))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=-1, right=len(ave_grads))
        # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.yscale('log')
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)],
                   ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.tight_layout()

        if show:
            matplotlib.use('TkAgg')
            plt.show()
            plt.close()
        else:
            fig_img = MiscUtils.fig2img(fig)

            plt.close()
            return fig_img

    @staticmethod
    def viz_sequence(data, viz=False):
        """Visualize a sequence of images

        Args:
            data: ndarray of shape (T, H, W, 3) or (T, H, W). Can be the output
                of MiscUtils.deprocess_rgb(). Can be a list of ndarray, in
                which case the figure will have multiple row.
            viz: whether to show the visualization (using TkAgg backend)

        Return:
            fig: figure handler
        """
        # Convert data to list if needed
        if isinstance(data, list):
            n_items = len(data)
        else:
            data = [data]
            n_items = 1

        # Get the number of frames
        n_frames = max([item.shape[0] for item in data])

        # Prepare figures
        fig, axes = plt.subplots(n_items, n_frames, figsize=(2*n_frames, 2*n_items))
        if n_items == 1:
            axes = [axes]

        # Plot frames
        for i, item in enumerate(data):
            assert isinstance(item, np.ndarray), 'Each item must be ndarray'
            assert item.ndim in [3, 4], 'Unsupported tensor dimensionality'

            # Starting offset in case sequences have different length
            offset = n_frames - item.shape[0]

            # Normalize if single channel
            if item.ndim == 3:
                vmin, vmax = item.min(), item.max()
            else:
                vmin, vmax = 0, 255

            for t in range(offset, n_frames):
                axes[i][t].imshow(item[t-offset], vmin=vmin, vmax=vmax)
                axes[i][t].set_xticks([])
                axes[i][t].set_yticks([])
        fig.tight_layout()

        if viz:
            matplotlib.use('TkAgg')
            plt.show()
        return fig

    @staticmethod
    def extend_path(path, root, check_fn=None):
        """Extend a given path using the root so that the returned path is root/path

        Args:
            path: the path to extend
            root: the root to extend path
            check_fn: function to validate path, e.g. os.path.isdir or
                os.path.isfile. If None, will not check the path

        Return:
            Extended path
        """
        if (check_fn is not None) and (not check_fn(path)):
            return os.path.join(root, path)
        return path

    @staticmethod
    def get_samples_from_loader(data_loader, indices):
        """Get samples from dataloader, given some specific indices. The loader
        must return pair of (sample, target), where sample is a dictionary of
        tensors and target is a tensor.
        """
        dataset = data_loader.dataset
        keys = dataset[0][0].keys()

        sample = {}
        for k in keys:
            sample[k] = torch.stack([dataset[i][0][k] for i in indices], dim=0)
        target = torch.stack([dataset[i][1] for i in indices], dim=0)
        return sample, target
