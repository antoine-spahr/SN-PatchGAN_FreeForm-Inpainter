"""
author: Antoine Spahr

date : 29.09.2020

----------

TO DO :
"""
import os
#import pandas as pd
import numpy as np
import skimage.io as io
import skimage
import cv2
import torch
from torch.utils import data

import src.dataset.transforms as tf

class InpaintDataset(data.Dataset):
    """
    Dataset object to load the data for the inpainting task.
    """
    def __init__(self, fn_list, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], output_size=256,
                 ff_param=dict(n_draw=(1,4), vertex=(15,30), brush_width=(15,25), angle=(0.5,2), length=(15,50),
                               n_salt_pepper=(0,15), salt_peper_radius=(1,6))):
        """
        Build a dataset Free Form inpainting from a image folder.
        ----------
        INPUT
            |---- fn_list (list of str) list of image file name (for format support see skimage.io.imread).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- output_size (int) the dimension of the output (H = W).
            |---- n_draw (tuple (low, high)) range of number of inpaint element to draw.
            |---- ff_param (dict) parameters for the free form mask generation. valid keys are:
            |        |---- vertex (tuple (low, high)) range of number of vertex for each inpaint element.
            |        |---- brush_width (tuple (low, high)) range of brush size to draw each inpaint element.
            |        |---- angle (tuple (low, high)) the range of angle between each vertex of an inpaint element. Note that every
            |        |               two segment, Pi is added to the angle to keep the drawing in the vicinity. Angle in radian.
            |        |---- length (tuple (low, high)) range of length for each segment.
            |        |---- n_salt_pepper (tuple (low, high)) range of number of salt and pepper disk element to draw. Set to (0,1)
            |        |               for no salt and pepper elements.
            |        |---- salt_peper_radius (tuple (low, high)) range of radius for the salt and pepper disk element.
        OUTPUT
            |---- Inpaint_dataset (torch.Dataset)
        """
        super(InpaintDataset, self).__init__()
        self.img_fn = fn_list
        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size),
                                    tf.ToTorchTensor())
        self.ff_param = ff_param

    def __len__(self):
        """
        eturn the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return len(self.img_fn)

    def __getitem__(self, idx):
        """
        Extract the CT sepcified by idx.
        ----------
        INPUT
            |---- idx (int) the sample index in self.data_df.
        OUTPUT
            |---- im (torch.tensor) the CT image with dimension (1 x H x W).
            |---- mask (torch.tensor) the inpaining mask with dimension (1 x H x W).
        """
        # load image and convert it in float in [0,1]
        im = skimage.img_as_float(io.imread(self.img_fn[idx]))
        # transform image
        im = self.transform(im)
        # get a random mask
        mask = self.random_ff_mask((im.shape[1], im.shape[2]), **self.ff_param)

        return im, tf.ToTorchTensor()(mask)

    @staticmethod
    def random_ff_mask(shape, n_draw=(1,4), vertex=(15,30), brush_width=(15,25), angle=(0.5,2), length=(15,50),
                  n_salt_pepper=(0,15), salt_peper_radius=(1,6)):
        """
        Generate a random inpainting mask with given shape.
        ----------
        INPUT
            |---- shape (tuple (h,w)) the size of the inpainting mask.
            |---- n_draw (tuple (low, high)) range of number of inpaint element to draw.
            |---- vertex (tuple (low, high)) range of number of vertex for each inpaint element.
            |---- brush_width (tuple (low, high)) range of brush size to draw each inpaint element.
            |---- angle (tuple (low, high)) the range of angle between each vertex of an inpaint element. Note that every
            |               two segment, Pi is added to the angle to keep the drawing in the vicinity. Angle in radian.
            |---- length (tuple (low, high)) range of length for each segment.
            |---- n_salt_pepper (tuple (low, high)) range of number of salt and pepper disk element to draw. Set to (0,1)
            |               for no salt and pepper elements.
            |---- salt_peper_radius (tuple (low, high)) range of radius for the salt and pepper disk element.
        OUTPUT
            |---- mask (np.array) the inpainting mask with value 1 on region to inpaint and zero otherwise.
        """
        h, w = shape
        mask = np.zeros(shape)
        # draw random number of patches
        for _ in range(np.random.randint(low=n_draw[0], high=n_draw[1])):
            n_vertex = np.random.randint(low=vertex[0], high=vertex[1])
            bw = np.random.randint(low=brush_width[0], high=brush_width[1])
            start_x, start_y = int(np.random.normal(w/2, w/4)), int(np.random.normal(h/2, h/4))
            #start_x, start_y = np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)

            beta = np.random.uniform(low=0, high=6.28)
            for i in range(n_vertex):
                alpha = beta + np.random.uniform(low=angle[0], high=angle[1])
                l = np.random.randint(low=length[0], high=length[1])
                if i % 2 == 0:
                    alpha = np.pi + alpha #2 * np.pi - angle # reverse mode
                # draw line
                end_x = (start_x + l * np.sin(alpha)).astype(np.int32)
                end_y = (start_y + l * np.cos(alpha)).astype(np.int32)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, bw)
                # set new start point
                start_x, start_y = end_x, end_y

        # salt and pepper
        for _ in range(np.random.randint(low=n_salt_pepper[0], high=n_salt_pepper[1])):
            start_x, start_y = np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)
            r = np.random.randint(low=salt_peper_radius[0], high=salt_peper_radius[1])
            cv2.circle(mask, (start_x, start_y), r, 1.0, -1)

        return mask

class ImgMaskDataset(data.Dataset):
    """
    Dataset object to load an image and mask together.
    """
    def __init__(self, img_fn, mask_fn, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], output_size=256):
        """
        Build a dataset for loading image and mask.
        ----------
        INPUT
            |---- img_fn (list of str) list of image file name (for format support see skimage.io.imread).
            |---- mask_fn (list of str) list of binary mask file image.
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- output_size (int) the dimension of the output (H = W).
        OUTPUT
            |---- RSNA_Inpaint_dataset (torch.Dataset) the RSNA dataset for inpainting.
        """
        super(ImgMaskDataset, self).__init__()
        self.mask_fn = mask_fn
        self.img_fn = img_fn
        assert len(self.mask_fn) == len(self.img_fn), f"The number of masks and image must be similar. \
                                                        Given {len(self.mask_fn)} masks and {len(self.img_fn)} images."

        self.transform = tf.Compose(tf.Resize(H=output_size, W=output_size),
                                    *augmentation_transform,
                                    tf.ToTorchTensor())

    def __len__(self):
        """
        eturn the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return len(self.img_fn)

    def __getitem__(self, idx):
        """
        Extract the image and mask sepcified by idx.
        ----------
        INPUT
            |---- idx (int) the sample index in self.data_df.
        OUTPUT
            |---- im (torch.tensor) the image with dimension (1 x H x W).
            |---- mask (torch.tensor) the mask with dimension (1 x H x W).
        """
        # load dicom and recover the CT pixel values
        im = skimage.img_as_float(io.imread(self.img_fn[idx]))
        mask = skimage.img_as_bool(io.imread(self.mask_fn[idx]))
        # transform image
        im, mask = self.transform(im, mask)

        return im, mask.float()
