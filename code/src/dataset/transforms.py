"""
author: Antoine Spahr

date : 29.09.2020

----------

TO DO :
"""
import torchvision.transforms
import torch
import math
import random
import numpy as np
import skimage.transform
import skimage.filters
import scipy.ndimage

class Compose:
    """
    Compose, in a sequential fashion, multiple transforms than can handle an
    image and a mask in their __call__.
    """
    def __init__(self, *transformations):
        """
        Build a sequential composition of transforms.
        ----------
        INPUT
            |---- *transformations (args) list of transforms to compose.
        OUTPUT
            |---- Transformation
        """
        self.transformations = transformations

    def __call__(self, image, mask=None):
        """
        Passes the image (and mask) through all the transforms.
        ----------
        INPUT
            |---- image () the input image.
            |---- mask () the input mask.
        OUTPUT
            |---- image () the transformed image.
            |---- mask () the transformed mask.
        """
        if mask is not None:
            for f in self.transformations:
                image, mask = f(image, mask)
            return image, mask
        else:
            for f in self.transformations:
                image = f(image)
            return image

    def __str__(self):
        """
        Transform printing format
        """
        tf_names = [str(t) for t in self.transformations]
        max_size = max(len(x) for x in tf_names)
        link = '\n' + '|'.center(max_size) + '\n' + 'V'.center(max_size) + '\n'
        return link.join([t.center(max_size) for t in tf_names])

    def __add__(self, other):
        """
        Define how to sum composition.
        """
        return Compose(*(self.transformations + other.transformations))

class Resize:
    """
    Resize the image and mask to the given height and width.
    """
    def __init__(self, H=256, W=256, mode='reflect'):
        """
        Build a to resize transform.
        ----------
        INPUT
            |---- H (int) the new Height.
            |---- W (int) the new Width.
        OUTPUT
            |---- Rotate transform.
        """
        self.H = H
        self.W = W
        self.mode = mode

    def __call__(self, image, mask=None):
        """
        Resize the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        image = skimage.transform.resize(image, (self.H, self.W, *image.shape[2:]), order=1, mode=self.mode)#, cval=image.min())
        if mask is not None:
            mask = skimage.transform.resize(mask, (self.H, self.W, *mask.shape[2:]), order=0, preserve_range=True)
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Resize(H={self.H}, W={self.W}, mode={self.mode})"

class Translate:
    """
    Randomly translate the image and mask in a range of fraction.
    """
    def __init__(self, low=-0.1, high=0.1, mode='constant'):
        """
        Build a to translate transform.
        ----------
        INPUT
            |---- low (float) the lower bound of the translate fraction.
            |---- high (float) the upper bound of the translate fraction.
        OUTPUT
            |---- Translate transform.
        """
        self.low = low
        self.high = high
        self.mode = mode

    def __call__(self, image, mask=None):
        """
        Randomly translate the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- (mask) (np.array) the converted mask.
        """
        # randomly sample a shift in x and y. No shift in z
        shift = [np.random.uniform(low=image.shape[0]*self.low, high=image.shape[0]*self.high),
                 np.random.uniform(low=image.shape[1]*self.low, high=image.shape[1]*self.high)]
        shift += [0] * (image.ndim - 2)

        image = scipy.ndimage.shift(image, shift, order=1, mode=self.mode)#, cval=image.min())
        #image = np.stack([scipy.ndimage.shift(image[:,:,i], shift[:2], order=3) for i in range(image.shape[2])], axis=2)
        if mask is not None:
            mask = scipy.ndimage.shift(mask, shift[:2], order=0)
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Translate(low={self.low}, high={self.high}, mode={self.mode})"

class Scale:
    """
    Randomly scale the image and mask in a range of factor.
    """
    def __init__(self, low=0.9, high=1.1, mode='constant'):
        """
        Build a to scale transform.
        ----------
        INPUT
            |---- low (float) the lower bound of the fraction.
            |---- high (float) the upper bound of the fraction.
        OUTPUT
            |---- Scale transform.
        """
        self.low = low
        self.high = high
        self.mode = mode

    def __call__(self, image, mask=None):
        """
        Randomly scale the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        # randomly sample a scale in x and y. No scale in z
        scale_factor = np.random.uniform(low=self.low, high=self.high)
        scales = [scale_factor] * 2
        scales += [1] * (image.ndim - 2)


        adjust_h = abs((np.round(scale_factor*image.shape[0]) - image.shape[0]) / 2)
        adjust_w = abs((np.round(scale_factor*image.shape[1]) - image.shape[1]) / 2)

        adjust_list = [(int(np.floor(adjust_h)), int(np.ceil(adjust_h))),
                       (int(np.floor(adjust_w)), int(np.ceil(adjust_w)))] + [(0,0)] * (image.ndim - 2)

        if scale_factor >= 1:
            image = skimage.util.crop(scipy.ndimage.zoom(image, scales, order=1, mode=self.mode), adjust_list)
            #image = np.stack([skimage.util.crop(scipy.ndimage.zoom(image[:,:,i], scales[:2], order=3), adjust_list[:2]) for i in range(image.shape[2])], axis=2)
            if mask is not None:
                mask = skimage.util.crop(scipy.ndimage.zoom(mask, scales[:2], order=0), adjust_list[:2])
        else:
            #image = skimage.util.pad(scipy.ndimage.zoom(image, scales, order=3), adjust_list)#, constant_values=image.min())
            image = np.pad(scipy.ndimage.zoom(image, scales, order=1, mode=self.mode), adjust_list, mode=self.mode)
            #image = np.stack([np.pad(scipy.ndimage.zoom(image[:,:,i], scales[:2], order=3), adjust_list[:2]) for i in range(image.shape[2])], axis=2)
            if mask is not None:
                #mask = skimage.util.pad(scipy.ndimage.zoom(mask, scales, order=0), adjust_list)
                mask = np.pad(scipy.ndimage.zoom(mask, scales[:2], order=0), adjust_list[:2])

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Scale(low={self.low}, high={self.high}, mode={self.mode})"

class Rotate:
    """
    Randomly rotate the image and mask in a range of angles.
    """
    def __init__(self, low=-10, high=10, mode='constant'):
        """
        Build a to rotate transform.
        ----------
        INPUT
            |---- low (float) the lower bound of the angle in degree.
            |---- high (float) the upper bound of the angle in degree.
        OUTPUT
            |---- Rotate transform.
        """
        self.low = low
        self.high = high
        self.mode = mode

    def __call__(self, image, mask=None):
        """
        Randomly rotate the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        # randomly sample an angle
        angle = np.random.uniform(low=self.low, high=self.high)

        image = scipy.ndimage.rotate(image, angle, axes=(1,0), order=1, reshape=False, mode=self.mode)#, cval=image.min())
        #image = np.stack([scipy.ndimage.rotate(image[:,:,i], angle, axes=(1,0), order=1, reshape=False) for i in range(image.shape[2])], axis=2)
        if mask is not None:
            mask = scipy.ndimage.rotate(mask, angle, axes=(1,0), order=0, reshape=False)
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Rotate(low={self.low}, high={self.high}, mode={self.mode})"

class HFlip:
    """
    Randomly flip the image and mask.
    """
    def __init__(self, p=0.5):
        """
        Build a horizontal-flip transform.
        ----------
        INPUT
            |---- p (float) probability of flipping. Must be in [0,1]
        OUTPUT
            |---- H-Flip tranform.
        """
        assert p >= 0 or p <= 1, f'Probability must be between 0 and 1. Given : {p}'
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randomly flip horizontally the np.array image and mask.
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        if np.random.random() < self.p:
            image = np.array(np.flip(image, axis=1)) # reconvert in array to make it contiguous
            if mask is not None:
                mask = np.array(np.flip(mask, axis=1))

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"HFlip(p={self.p})"

class VFlip:
    """
    Randomly flip the image and mask.
    """
    def __init__(self, p=0.5):
        """
        Build a vertical-flip transform.
        ----------
        INPUT
            |---- p (float) probability of flipping. Must be in [0,1]
        OUTPUT
            |---- V-Flip tranform.
        """
        assert p >= 0 or p <= 1, f'Probability must be between 0 and 1. Given : {p}'
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randomly flip vertically the np.array image and mask.
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        if np.random.random() < self.p:
            image = np.array(np.flip(image, axis=0)) # reconvert in array to make it contiguous
            if mask is not None:
                mask = np.array(np.flip(mask, axis=0))

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"VFlip(p={self.p})"

class GaussianBlur:
    """
    Randomly apply a gaussian blurring with random std.
    """
    def __init__(self, p=0.5, sigma=(0.1, 2.0)):
        """
        Build a to rotate transform.
        ----------
        INPUT
            |---- p (float) probability of blurring.
            |---- sigma (tuple) range of std.
        OUTPUT
            |---- Gaussian Blur transform
        """
        assert p >= 0 or p <= 1, f'Probability must be between 0 and 1. Given : {p}'
        self.p = p
        self.sigma = sigma

    def __call__(self, image, mask=None):
        """
        Randomly blur the input image with a std sampled uniformly from the range of sigma. If provided, a mask is simply
        passed without transformation.
        ----------
        INPUT
            |---- image (PIL.Image) the image to adjust.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (np.array) the adjusted image.
            |---- mask (np.array) the passed mask.
        """
        if np.random.random() < self.p:
            s = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
            image = skimage.filters.gaussian(image, sigma=s)

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"GaussianBlur(sigma={self.sigma}, p={self.p})"

class AdjustBrightness:
    """
    Randomly adjust brighness of image.
    """
    def __init__(self, p=0.5, low=-0.3, high=0.2):
        """
        Build a brighness adjustment transform.
        ----------
        INPUT
            |---- p (float) probability of brighness adjustment.
            |---- low (float) lower bound of the brighness factor. (negative : reduced brighness, positive : increase brightness).
            |---- high (float) upper bound of the brighness factor. (negative : reduced brighness, positive : increase brightness).
        OUTPUT
            |---- Brightness adjust tranform.
        """
        assert p >= 0 or p <= 1, f'Probability must be between 0 and 1. Given : {p}'
        assert low <= high, f'Low bounds must be lower than high bounds : {low} !< {high}'
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randomly adjust brighness of the np.array image. A mask is just passed if given. The image range is expected to be [0,1].
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        if np.random.random() < self.p:
            factor = np.random.uniform(low=self.low, high=self.high)
            image = image + factor
            image = np.clip(image, 0.0, 1.0)

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"AdjustBrightness(p={self.p}, low={self.low}, high={self.high})"

class AdjustContrast:
    """
    Randomly adjust contrast of image.
    """
    def __init__(self, p=0.5, low=0.5, high=1.5):
        """
        Build a brighness adjustment transform.
        ----------
        INPUT
            |---- p (float) probability of contrast adjustment.
            |---- low (float) lower bound of the contrast factor. (<1 : reduced contrast, >1 : increase brightness).
            |---- high (float) upper bound of the contrast factor. (<1 : reduced brighness, >1 : increase brightness).
        OUTPUT
            |---- Contrast adjust tranform.
        """
        assert p >= 0 or p <= 1, f'Probability must be between 0 and 1. Given : {p}'
        assert low <= high, f'Low bounds must be lower than high bounds : {low} !< {high}'
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randomly adjust contrast of the np.array image. A mask is just passed if given. The image range is expected to be [0,1].
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        if np.random.random() < self.p:
            factor = np.random.uniform(low=self.low, high=self.high)
            image = image * factor
            image = np.clip(image, 0.0, 1.0)

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"AdjustContrast(p={self.p}, low={self.low}, high={self.high})"

class RandomCropResize:
    """
    Randomly crop and resize the image.
    """
    def __init__(self, crop_scales=(0.2, 1.0), crop_ratios=(3./4., 4./3.)):
        """
        Build a to rotate transform.
        ----------
        INPUT
            |---- crop_scales (tuple: (low, high)) the range of possible scale of the crop.
            |---- crop_ratios (tuple: (low, high)) the range of possible aspect ratio in the crop.
        OUTPUT
            |---- Random Crop and Resize transform
        """
        assert crop_scales[1] <= 1, f'The upper crop scale bound cannot be above 1. Given {crop_scales[1]}'
        self.crop_scales = crop_scales
        self.crop_ratios = crop_ratios

    def __call__(self, image, mask=None):
        """
        Randomly crop and resize the input image and mask.
        ----------
        INPUT
            |---- image (PIL.Image) the image to adjust.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (np.array) the adjusted image.
            |---- mask (np.array) the passed mask.
        """
        # get crop parameter
        image_size = image.shape
        # scale = np.random.uniform(low=self.crop_scales[0], high=self.crop_scales[1])
        # h, w = int(scale * image.shape[0]), int(scale * image.shape[1])
        # # sample crop position
        # i, j = np.random.randint(low=0, high=image.shape[0] - h), np.random.randint(low=0, high=image.shape[1] - w)
        i, j, h, w = self.get_params(image_size, self.crop_scales, self.crop_ratios)
        # crop & resize
        image = skimage.transform.resize(image[i:i+h, j:j+w], (image_size[0], image_size[1], *image_size[2:]), order=1)

        if mask is not None:
            mask = skimage.transform.resize(mask[i:i+h, j:j+w], (mask[0], image_size[1], *image_size[2:]), order=0)
            return image, mask
        else:
            return image

    def get_params(self, img_size, scale, ratio):
        """
        Compute position and size of crop. (method from torchvision).
        ----------
        INPUT
            |---- img_size (tuple (h,w)) the image size.
            |---- scale (tuple) range of size of the origin size cropped
            |---- ratio (tuple) range of aspect ratio of the origin aspect ratio cropped
        OUTPUT
            |---- params (tuple :(i, j, h, w)) crop position and size.
        """
        height, width = img_size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomCropResize(crop_scales={self.crop_scales}, crop_ratios={self.crop_ratios})"

class ToTorchTensor:
    """
    Convert the image (and mask) to a torch.Tensor.
    """
    def __init__(self):
        """
        Build a to tensor transform.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- ToTensor transform.
        """

    def __call__(self, image, mask=None):
        """
        Convert the image and mask (np.array) to a torch.Tensor
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (torch.Tensor) the converted image.
            |---- mask (torch.Tensor) the converted mask.
        """
        image = torchvision.transforms.ToTensor()(image)
        if mask is not None:
            mask = torchvision.transforms.ToTensor()(mask)
            return image, mask.bool()
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return "ToTorchTensor()"
