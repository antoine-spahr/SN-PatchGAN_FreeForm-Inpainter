"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :
"""
import torch
import torch.nn as nn
import numpy as np

class DiscountedL1(nn.Module):
    """
    Pytorch implementation of the Discounted L1 Loss proposed in Yu et al. 2018 (Generative Inpainting with Contextual
    Attention). It weight a L1 loss on a mask by the distance to the closest element not of the mask.
    """
    def __init__(self, gamma=0.99, reduction='mean', device='cuda'):
        """
        Build a DiscountedL1 Loss module.
        ----------
        INPUT
            |---- gamma (float) hyperparameter for the weight computation. The weight of a position is given by gamma ** dist.
            |---- reduction (str) the reduction to apply. One of 'mean', 'sum', 'none'.
            |---- device (str) the device on which to operate.
        OUTPUT
            |---- DiscountedL1 (nn.Module) the discounted L1 loss module.
        """
        super(DiscountedL1, self).__init__()
        self.gamma = torch.tensor(gamma, device=device)
        self.L1 = nn.L1Loss(reduction='none')
        assert reduction in ['mean', 'none', 'sum'], f"Reduction mode: '{reduction}' is not supported. Use either 'mean', 'sum' or 'none'."
        self.reduction = reduction
        self.device = device

    def forward(self, rec, im, mask):
        """
        Forward pass of the L1 Discounted loss. Note that a weight of 1.0 is used outside the mask (i.e. where mask = 0).
        ----------
        INPUT
            |---- rec (torch.tensor) the reconstructed image as (B x C x H x W).
            |---- im (torch.tensor) the reference image as (B x C x H x W).
            |---- mask (torch.tensor) the mask where to L1 is weighted as (B x 1 x H x W).
        """
        #assert rec.shape == im.shape, f'Shape mismatch between image and reconstruction: {rec.shape} vs {im.shape}'
        l1_loss = self.L1(rec, im)
        weight = (self.gamma.view(1,1,1,1) ** self.get_dist_mask(mask)) * mask # consider only pixel on mask and give more inportance of one close to border
        l1_loss = l1_loss * weight
        # Apply the reduction
        if self.reduction == 'mean':
            return l1_loss.mean()
        elif self.reduction == 'sum':
            return l1_loss.sum()
        elif self.reduction == 'none':
            return l1_loss

    def get_dist_mask(self, mask):
        """
        Compute the distance map of the given mask. Each pixel of the mask is assigned the minimum euclidian distance to
        a pixel with value 0.
        ----------
        INPUT
            |---- mask (torch.tensor) the mask where to L1 is weighted as (B x 1 x H x W).
        OUTPUT
            |---- dist_mask (torch.tensor) Same size tensor as mask with value of distance to closest 0 element.
        """
        # get mask of object contours (= dilation of mask - mask)
        border = nn.functional.max_pool2d(mask, 3, stride=1, padding=1, dilation=1) - mask
        dist_mask = []
        for i in range(mask.shape[0]):
            # compute distance between mask element and every non-mask border element
            dst = torch.cdist(torch.nonzero(mask[i,0,:,:]).unsqueeze(0).float(), torch.nonzero(border[i,0,:,:]).unsqueeze(0).float(), p=2)
            # fill every mask position with the minimum distance (euclidian)
            msk = torch.zeros(mask.shape[2:], device=self.device)
            if dst.shape[2] > 0:
                msk[torch.split(torch.nonzero(mask[i,0,:,:]).t(), 1, dim=0)] = dst.min(dim=2)[0]
            dist_mask.append(msk.view(1,1,*msk.shape))

        return torch.cat(dist_mask, dim=0)
