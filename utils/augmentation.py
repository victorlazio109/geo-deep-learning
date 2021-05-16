# WARNING: data being augmented may be scaled to (0,1) rather, for example, (0,255).
#          Therefore, implementing radiometric
# augmentations (ex.: changing hue, saturation, brightness, contrast) may give undesired results.
# Scaling process is done in images_to_samples.py l.215
import numbers
import math
import warnings
from typing import Sequence

import torch
# import torch should be first.
# Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import random
import numpy as np
from skimage import transform, exposure
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from utils.utils import get_key_def, pad, minmax_scale, BGR_to_RGB


def compose_transforms(params, dataset, type='', ignore_index=None):
    """
    Function to compose the transformations to be applied on every batches.
    :param params: (dict) Parameters found in the yaml config file
    :param dataset: (str) One of 'trn', 'val', 'tst'
    :param type: (str) One of 'geometric', 'radiometric'
    :return: (obj) PyTorch's compose object of the transformations to be applied.
    """
    lst_trans = []
    input_space = get_key_def('BGR_to_RGB', params['global'], False)
    scale = get_key_def('scale_data', params['global'], None)
    norm_mean = get_key_def('mean', params['training']['normalization'])
    norm_std = get_key_def('std', params['training']['normalization'])
    random_radiom_trim_range = get_key_def('random_radiom_trim_range', params['training']['augmentation'], None)

    if dataset == 'trn':

        if type == 'radiometric':
            noise = get_key_def('noise', params['training']['augmentation'], None)

            if random_radiom_trim_range:  # Contrast stretching
                # FIXME: test this. Assure compatibility with CRIM devs (don't trim metadata)
                lst_trans.append(RadiometricTrim(random_range=random_radiom_trim_range))

            if noise:
                raise NotImplementedError

        elif type == 'geometric':
            geom_scale_range = get_key_def('geom_scale_range', params['training']['augmentation'], None)
            hflip = get_key_def('hflip_prob', params['training']['augmentation'], None)
            rotate_prob = get_key_def('rotate_prob', params['training']['augmentation'], None)
            rotate_limit = get_key_def('rotate_limit', params['training']['augmentation'], None)
            crop_size = get_key_def('target_size', params['training'], None)

            if geom_scale_range:  # TODO: test this.
                lst_trans.append(GeometricScale(range=geom_scale_range))

            if hflip:
                lst_trans.append(HorizontalFlip(prob=params['training']['augmentation']['hflip_prob']))

            if rotate_limit and rotate_prob:
                lst_trans.append(
                    RandomRotationTarget(
                        limit=rotate_limit, prob=rotate_prob, ignore_index=ignore_index
                    )
                )

            if crop_size:
                lst_trans.append(RandomCrop(sample_size=crop_size, ignore_index=ignore_index))

    if type == 'totensor':
        # Contrast stretching at eval. Use mean of provided range
        if not dataset == 'trn' and random_radiom_trim_range:
            # Assert range is number or 2 element sequence
            RadiometricTrim.input_checker(random_radiom_trim_range)
            if isinstance(random_radiom_trim_range, numbers.Number):
                trim_at_eval = random_radiom_trim_range
            else:
                trim_at_eval = round((random_radiom_trim_range[-1] - random_radiom_trim_range[0]) / 2, 1)
            lst_trans.append(RadiometricTrim(random_range=[trim_at_eval, trim_at_eval]))

        if input_space:
            lst_trans.append(BgrToRgb(input_space))

        if scale:
            lst_trans.append(Scale(scale))  # TODO: assert coherence with below normalization
            # lst_trans.append(VegetationIndex())

        if norm_mean and norm_std:
            lst_trans.append(Normalize(mean=params['training']['normalization']['mean'],
                                       std=params['training']['normalization']['std']))

        lst_trans.append(ToTensorTarget(
            num_classes=params['global']['num_classes']))  # Send channels first, convert numpy array to torch tensor

    return transforms.Compose(lst_trans)


class RadiometricTrim(object):
    """Trims values left and right of the raster's histogram. Also called linear scaling or enhancement.
    Percentile, chosen randomly based on inputted range, applies to both left and right sides of the histogram.
    Ex.: Values below the 1.7th and above the 98.3th percentile will be trimmed if random value is 1.7"""
    def __init__(self, random_range):
        """
        @param random_range: numbers.Number (float or int) or Sequence (list or tuple) with length of 2
        """
        random_range = self.input_checker(random_range)
        self.range = random_range

    @staticmethod
    def input_checker(input_param):
        if not isinstance(input_param, (numbers.Number, Sequence)):
            raise TypeError('Got inappropriate range arg')

        if isinstance(input_param, Sequence) and len(input_param) != 2:
            raise ValueError(f"Range must be an int or a 2 element tuple or list, "
                             f"not a {len(input_param)} element {type(input_param)}.")

        if isinstance(input_param, numbers.Number):
            input_param = [input_param, input_param]
        return input_param

    def __call__(self, sample):
        # Choose trimming percentile withing inputted range
        trim = round(random.uniform(self.range[0], self.range[-1]), 1)
        # Determine output range from datatype
        out_dtype = sample['metadata']['dtype']
        # Create empty array with shape of input image
        rescaled_sat_img = np.empty(sample['sat_img'].shape, dtype=sample['sat_img'].dtype)
        # Loop through bands
        for band_idx in range(sample['sat_img'].shape[2]):
            band = sample['sat_img'][:, :, band_idx]
            band_histogram = sample['metadata']['source_raster_bincount'][f'band{band_idx}']
            # Determine what is the index of nonzero pixel corresponding to left and right trim percentile
            sum_nonzero_pix_per_band = sum(band_histogram)
            left_pixel_idx = round(sum_nonzero_pix_per_band / 100 * trim)
            right_pixel_idx = round(sum_nonzero_pix_per_band / 100 * (100-trim))
            cumulative_pixel_count = 0
            # TODO: can this for loop be optimized? Also, this hasn't been tested with non 8-bit data. Should be fine though.
            # Loop through pixel values of given histogram
            for pixel_val, count_per_pix_val in enumerate(band_histogram):
                lower_limit = cumulative_pixel_count
                upper_limit = cumulative_pixel_count + count_per_pix_val
                # Check if left and right pixel indices are contained in current lower and upper pixels count limits
                if lower_limit <= left_pixel_idx <= upper_limit:
                    left_pix_val = pixel_val
                if lower_limit <= right_pixel_idx <= upper_limit:
                    right_pix_val = pixel_val
                cumulative_pixel_count += count_per_pix_val
            # Enhance using above left and right pixel values as in_range
            rescaled_band = exposure.rescale_intensity(band, in_range=(left_pix_val, right_pix_val), out_range=out_dtype)
            # Write each enhanced band to empty array
            rescaled_sat_img[:, :, band_idx] = rescaled_band
        sample['sat_img'] = rescaled_sat_img
        return sample


class Scale(object):
    """
    Scale array values from range [0,255]  or [0,65535] to values in config ([0,1] or [-1, 1])
    Guidelines for pre-processing: http://cs231n.github.io/neural-networks-2/#datapre
    """
    def __init__(self, range):
        if isinstance(range, Sequence) and len(range) == 2:
            self.sc_min = range[0]
            self.sc_max = range[1]
        else:
            raise TypeError('Got inappropriate scale arg')

    @staticmethod
    def range_values_raster(raster, dtype):
        min_val, max_val = np.nanmin(raster), np.nanmax(raster)
        if 'int' in dtype:
            orig_range = (np.iinfo(dtype).min, np.iinfo(dtype).max)
        elif min_val >= 0 and max_val <= 255:
            orig_range = (0, 255)
            warnings.warn(f"Values in input image of shape {raster.shape} "
                          f"range from {min_val} to {max_val}."
                          f"Image will be considered 8 bit for scaling.")
        elif min_val >= 0 and max_val <= 65535:
            orig_range = (0, 65535)
            warnings.warn(f"Values in input image of shape {raster.shape} "
                          f"range from {min_val} to {max_val}."
                          f"Image will be considered 16 bit for scaling.")
        else:
            raise ValueError(f"Invalid values in input image. They should range from 0 to 255 or 65535, not"
                             f"{min_val} to {max_val}.")
        return orig_range


    def __call__(self, sample):
        """
        Args:
            sample (ndarray): Image to be scaled.

        Returns:
            ndarray: Scaled image.
        """
        out_dtype = sample['metadata']['dtype']
        orig_range = self.range_values_raster(sample['sat_img'], out_dtype)
        sample['sat_img'] = minmax_scale(img=sample['sat_img'], orig_range=orig_range, scale_range=(self.sc_min, self.sc_max))

        return sample


class GeometricScale(object):
    """Randomly resize image according to a certain range."""
    def __init__(self, range):
        self.range = range

    def __call__(self, sample):
        scale_factor = round(random.uniform(range[0], range[-1]), 1)
        output_width = sample['sat_img'].shape[0] * scale_factor
        output_height =  sample['sat_img'].shape[1] * scale_factor
        sat_img = transform.resize(sample['sat_img'], output_shape=(output_height, output_width))
        map_img = transform.resize(sample['map_img'], output_shape=(output_height, output_width))
        # skel_img = transform.resize(sample['skel_img'], output_shape=(output_height, output_width))
        sample['sat_img'] = sat_img
        sample['map_img'] = map_img
        # sample['skel_img'] = skel_img
        return sample


class RandomRotationTarget(object):
    """Rotate the image and target randomly."""
    def __init__(self, limit, prob, ignore_index):
        self.limit = limit
        self.prob = prob
        self.ignore_index = ignore_index

    def __call__(self, sample):
        if random.random() < self.prob:
            angle = np.random.choice([90, 180, 270])
            sat_img = transform.rotate(sample['sat_img'], angle, preserve_range=True, cval=np.nan)
            map_img = transform.rotate(sample['map_img'], angle, preserve_range=True, order=0, cval=self.ignore_index)
            # skel_img = transform.rotate(sample['skel_img'], angle, preserve_range=True, order=0, cval=self.ignore_index)
            sample['sat_img'] = sat_img
            sample['map_img'] = map_img
            # sample['skel_img'] = skel_img
            return sample
        else:
            return sample


class HorizontalFlip(object):
    """Flip the input image and reference map horizontally, with a probability."""
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sat_img = np.ascontiguousarray(sample['sat_img'][:, ::-1, ...])
            map_img = np.ascontiguousarray(sample['map_img'][:, ::-1, ...])
            # skel_img = np.ascontiguousarray(sample['skel_img'][:, ::-1, ...])
            sample['sat_img'] = sat_img
            sample['map_img'] = map_img
            # sample['skel_img'] = skel_img
        return sample


class RandomCrop(object):  
    # TODO: what to do with overlap in samples_prep (images_to_samples, l.106)?
    #       overlap doesn't need to be larger than, say, 5%
    """Randomly crop image according to a certain dimension.
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop
    to support >3 band images (not currently supported by PIL)"""
    def __init__(self, sample_size, padding=3, pad_if_needed=True, ignore_index=0):
        if isinstance(sample_size, numbers.Number):
            self.size = (int(sample_size), int(sample_size))
        else:
            self.size = sample_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.ignore_index = ignore_index

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        """
        Args:
            sample (ndarray): Image to be cropped.

        Returns:
            ndarray: Cropped image.
        """
        sat_img = sample['sat_img']
        map_img = sample['map_img']
        # skel_img = sample['skel_img']

        if self.padding is not None:
            sat_img = pad(sat_img, self.padding, np.nan)  # Pad with nan values for sat_img
            map_img = pad(map_img, self.padding, self.ignore_index)  # Pad with dontcare values for map_img
            # skel_img = pad(map_img, self.padding, self.ignore_index)

        # pad the height if needed
        if self.pad_if_needed and sat_img.shape[0] < self.size[0]:
            sat_img = pad(sat_img, (0, self.size[0] - sat_img.shape[0]), np.nan)
        # pad the width if needed
        if self.pad_if_needed and sat_img.shape[1] < self.size[1]:
            sample = pad(sat_img, (self.size[1] - sat_img.shape[1], 0), np.nan)

        # pad the height if needed
        if self.pad_if_needed and map_img.shape[0] < self.size[0]:
            map_img = pad(map_img, (0, self.size[0] - map_img.shape[0]), self.ignore_index)
        # pad the width if needed
        if self.pad_if_needed and map_img.shape[1] < self.size[1]:
            map_img = pad(map_img, (self.size[1] - map_img.shape[1], 0), self.ignore_index)

        # pad the height if needed
        # if self.pad_if_needed and skel_img.shape[0] < self.size[0]:
        #     skel_img = pad(skel_img, (0, self.size[0] - skel_img.shape[0]), self.ignore_index)
        # pad the width if needed
        # if self.pad_if_needed and skel_img.shape[1] < self.size[1]:
        #     skel_img = pad(skel_img, (self.size[1] - skel_img.shape[1], 0), self.ignore_index)

        i, j, h, w = self.get_params(sat_img, self.size)

        sat_img = sat_img[i:i + h, j:j + w]
        map_img = map_img[i:i + h, j:j + w]
        # skel_img = skel_img[i:i + h, j:j + w]

        sample['sat_img'] = sat_img
        sample['map_img'] = map_img
        # sample['skel_img'] = skel_img
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class Normalize(object):
    """Normalize Image with Mean and STD and similar to Pytorch(transform.Normalize) function """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if self.mean or self.std != []:
            sat_img = (sample['sat_img'] - self.mean) / self.std
            sample['sat_img'] = sat_img
            return sample
        else:
            return sample

class BgrToRgb(object):
    """Normalize Image with Mean and STD and similar to Pytorch(transform.Normalize) function """

    def __init__(self, bgr_to_rgb):
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, sample):
        sat_img = BGR_to_RGB(sample['sat_img']) if self.bgr_to_rgb else sample['sat_img']
        sample['sat_img'] = sat_img

        return sample


class VegetationIndex(object):
    """Normalize Image with Mean and STD and similar to Pytorch(transform.Normalize) function """

    def __init__(self):
        pass

    def __call__(self, sample):
        sat_img = sample['sat_img']
        R = sat_img[:, :, 0]
        # G = sat_img[:, :, 1]
        B = sat_img[:, :, 2]
        N = sat_img[:, :, -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (N / 10000 - R / 10000) / (N / 10000 + R / 10000)
            # tdvi = 1.5 * ((N - R) / np.sqrt(N ** 2.0 + R + 0.5))
            # evi = 2.5 * (N - R) / (N + 6*R - 7.5*B + 1)

            evi = 2.5 * ((N / 10000 - R / 10000) / (N / 10000 + 6 * R / 10000 - 7.5 * B / 10000 + 1))

        sample['sat_img'] = minmax_scale(img=sat_img, orig_range=(0, 255), scale_range=(0, 1))
        # sample['sat_img'] = np.concatenate((sample['sat_img'], evi[:, :, np.newaxis]), axis=-1)
        sample['sat_img'] = np.concatenate((sample['sat_img'], ndvi[:, :, np.newaxis], evi[:, :, np.newaxis]), axis=-1)
        # print(sample['sat_img'].shape)

        return sample


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        sat_img = np.nan_to_num(sample['sat_img'], copy=False)
        sat_img = np.float32(np.transpose(sat_img, (2, 0, 1)))
        sat_img = torch.from_numpy(sat_img)

        map_img = None
        # skel_img = None
        if 'map_img' in sample.keys():
            if sample['map_img'] is not None:  # This can also be used in inference.
                map_img = np.int64(sample['map_img'])
                # skel_img = np.int64(sample['skel_img'])
                map_img[map_img > self.num_classes] = 0
                # skel_img[skel_img > self.num_classes] = 0
                map_img = torch.from_numpy(map_img)
                # skel_img = torch.from_numpy(skel_img)
        return {'sat_img': sat_img, 'map_img': map_img}


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
