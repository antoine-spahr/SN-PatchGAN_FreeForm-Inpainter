"""
author: Antoine Spahr

date : 16.11.2020

----------

TO DO :
"""
import os
import json
import logging
import time
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.io as io
from skimage import img_as_ubyte
from datetime import timedelta

from src.models.optim.LossFunctions import DiscountedL1
from src.utils.utils import print_progessbar

class SNPatchGAN:
    """
    Define a SN-PatchGAN model for image inpainting as proposed by Yu et al. in Generative Inpainting with Contextual
    Attention (2018) and Free Form inpainting with Gated Convolution (2018).
    """
    def __init__(self, generator, discriminator, n_epoch=100, batch_size=16, lr_g=1e-3, lr_d=1e-3,
                 lr_scheduler=optim.lr_scheduler.ExponentialLR, lr_scheduler_kwargs=dict(gamma=0.95), gammaL1=0.99,
                 lambda_L1=0.5, lambda_gan=0.5, weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False,
                 checkpoint_freq=3):
        """
        Build a SNPatchGAN model enabling to train and evaluate it together with utility functions.
        ----------
        INPUT
            |---- generator (nn.Module) the generator network. It must be an chain of two Encoder-Decoder-like network
            |               that takes an image and mask as input and return both the intermediate inpaint as well as the
            |               final inpaint image.
            |---- discriminator (nn.Module) the discriminator network. It must be a convolutional network taking an image
            |               and inpaint mask as input and output a feature map.
            |---- n_epoch (int) number of training epochs to perform
            |---- batch_size (int) batch_size for data-loading.
            |---- lr_g (float) the learning rate for the generator network optimization.
            |---- lr_d (float) the learning rate for the discriminator network optimization.
            |---- lr_scheduler (torch.optim.lr_scheduler) the learning rate evolution scheme to use for both optimization.
            |---- lr_scheduler_kwargs (dict) the keyword arguments to be passed to the lr_scheduler.
            |---- gammaL1 (float) the hyperparameter gamma of the discounted L1 loss. (see DiscountedL1 class)
            |---- lambda_L1 (float) the weight of the reconstruction loss in the generator's weight update.
            |---- lambda_gan (float) the weight of the GAN loss in the generator's weights update
            |---- weight_decay (float) L2-regularization of weight for the Adam Optimizer.
            |---- device (str) the device to use.
            |---- print_progess (bool) whether to print progress bar for batch processing.
            |---- checkpoint_freq (int) the frequency at which checkpoint are saved (in term of epoch).
        OUTPUT
            |---- SNPatchGAN () the SN-PatchGAN model for inpainting.
        """
        # network
        self.generator = generator
        self.generator = self.generator.to(device)
        self.discriminator = discriminator
        self.discriminator = self.discriminator.to(device)
        # data param
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        # optimization  param
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        # loss param
        self.gammaL1 = gammaL1
        self.lambda_L1 = lambda_L1
        self.lambda_gan = lambda_gan
        # other
        self.device = device
        self.print_progress = print_progress
        self.checkpoint_freq = checkpoint_freq
        # outputs
        self.outputs = {
            "train" : {
                "time": None,
                "evolution": None
            },
            "eval": None
        }

    def train(self, dataset, checkpoint_path=None, valid_dataset=None, valid_path=None, save_freq=5):
        """
        Train the SN-PatchGAN for the inpainting task on the given dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset from which data will be loaded. It has to return the
            |               original image the inpainting mask.
            |---- checkpoint_path (str) the path to load/save checkpoint from. If None, no checkpoints are saved/loaded.
            |---- valid_dataset (torch.utils.data.Dataset) dataset returning a small sample of fixed pairs (image, mask)
            |               on which to validate the GAN over training. It must return an image tensor of dimension
            |               [N_samp, C, H, W] and a mask tensor of dimension [N_samp, 1, H, W]. If None, no validation is
            |               performed during training.
            |---- valid_path (str) path to directory where to save the inpaint results of the valida_data as .png. If
            |               None the inpaint results are not saved.
            |---- save_freq (int) the frequency of epoch to save the validation inpainting results.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()
        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=True, worker_init_fn=lambda _: np.random.seed())
        # make optimizers
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, weight_decay=self.weight_decay, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, weight_decay=self.weight_decay, betas=(0.5, 0.999))
        # make scheduler
        scheduler_g = self.lr_scheduler(optimizer_g, **self.lr_scheduler_kwargs)
        scheduler_d = self.lr_scheduler(optimizer_d, **self.lr_scheduler_kwargs)
        # make L1 loss function
        L1Loss = DiscountedL1(gamma=self.gammaL1, reduction='mean', device=self.device)
        # Load checkpoint if present
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            n_epoch_finished = checkpoint['n_epoch_finished']
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
            scheduler_g.load_state_dict(checkpoint['lr_g_state'])
            scheduler_d.load_state_dict(checkpoint['lr_d_state'])
            epoch_loss_list = checkpoint['loss_evolution']
            logger.info(f'Checkpoint loaded with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            logger.info('No Checkpoint found. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = [] # Placeholder for epoch evolution

        # start training
        logger.info('Start training the inpainting SN-PatchGAN.')
        start_time = time.time()
        n_batch = len(loader)
        # train loop
        for epoch in range(n_epoch_finished, self.n_epoch):
            self.generator.train()
            self.discriminator.train()
            epoch_loss_l1, epoch_gan_loss_g, epoch_loss_d, epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(loader):
                #unpack data
                im, mask = data
                im = im.to(self.device).float().requires_grad_(True)
                mask = mask.to(self.device).float().requires_grad_(True)

                ### GENERATE FAKE
                fake_im2, fake_im1 = self.generator(im, mask)
                # keep only generated element on mask otherwise use original image
                fake_im2 = im * (1 - mask) + fake_im2 * mask
                fake_im1 = im * (1 - mask) + fake_im1 * mask

                ### TRAIN DISCRIMINATOR
                optimizer_d.zero_grad()
                # pass fake and real in discriminator
                fake_repr = self.discriminator(fake_im2.detach(), mask)
                real_repr = self.discriminator(im, mask)
                # compute discriminator hinge loss
                loss_d_real, loss_d_fake = torch.mean(F.relu(1.0 - real_repr)), torch.mean(F.relu(1.0 + fake_repr))
                loss_d = loss_d_real + loss_d_fake
                # update discriminator's weights
                loss_d.backward()
                optimizer_d.step()

                epoch_loss_d += loss_d.item()
                epoch_loss_d_real += loss_d_real.item()
                epoch_loss_d_fake += loss_d_fake.item()

                ### TRAIN GENERATOR
                optimizer_g.zero_grad()
                # compute discounted L1 reconstruction loss
                loss_l1 = L1Loss(fake_im1, im, mask) + L1Loss(fake_im2, im, mask)
                # compute the GAN loss with new forward pass through the updated Discriminator
                fake_repr = self.discriminator(fake_im2, mask)
                loss_gan = -torch.mean(fake_repr)
                loss_g = self.lambda_L1 * loss_l1 + self.lambda_gan * loss_gan
                # update generator's weights
                loss_g.backward()
                optimizer_g.step()

                # sum losses
                epoch_loss_g += loss_g.item()
                epoch_loss_l1 += loss_l1.item()
                epoch_gan_loss_g += loss_gan.item()

                # print batch summary --> state of one GAN iteration
                logger.info(f"| Epoch {epoch+1:03}/{self.n_epoch:03} | Batch {b+1:04}/{n_batch:04} "
                            f"| LossG {loss_g.item():.6f} -> L1 {loss_l1.item():.5f} +  GAN_G {loss_gan.item():.6f} "
                            f"| LossD {loss_d.item():.6f} -> L_real {loss_d_real.item():.6f} + L_fake {loss_d_fake.item():.6f} ")

            # Validation on set of fixed images
            if valid_dataset:
                save_path = valid_path if (epoch+1)%save_freq == 0 else None
                valid_l1 = self.validate(valid_dataset, save_path=save_path, epoch=epoch+1)

            # print epoch summary
            logger.info('|' + '-'*30 + f" Summary Epoch {epoch+1:03}/{self.n_epoch:03} " + '-'*30)
            logger.info(f"| Time {timedelta(seconds=int(time.time() - epoch_start_time))}")
            logger.info(f"| LossG {epoch_loss_g/n_batch:.6f} -> L1 {epoch_loss_l1/n_batch:.6f} + GAN_G {epoch_gan_loss_g/n_batch:.6f}")
            logger.info(f"| LossD {epoch_loss_d/n_batch:.6f} -> L_real {epoch_loss_d_real/n_batch:.6f} + L_fake {epoch_loss_d_fake/n_batch:.6f}")
            logger.info(f"| Valid L1 Loss {valid_l1:.6f}")
            logger.info(f"| lr_g {scheduler_g.get_last_lr()[0]:.6f} | lr_d {scheduler_d.get_last_lr()[0]:.6f} |")
            logger.info('|' + '-'*83)

            # store epoch
            epoch_loss_list.append([epoch+1, epoch_loss_l1/n_batch, epoch_gan_loss_g/n_batch, epoch_loss_g/n_batch, epoch_loss_d/n_batch, valid_l1])

            # update lr
            scheduler_d.step()
            scheduler_g.step()

            # save checkpoint
            if (epoch+1)%self.checkpoint_freq == 0 and checkpoint_path is not None:
                checkpoint ={
                    'n_epoch_finished': epoch+1,
                    'generator_state': self.generator.state_dict(),
                    'discriminator_state': self.discriminator.state_dict(),
                    'optimizer_g_state': optimizer_g.state_dict(),
                    'optimizer_d_state': optimizer_d.state_dict(),
                    'lr_g_state': scheduler_g.state_dict(),
                    'lr_d_state': scheduler_d.state_dict(),
                    'loss_evolution': epoch_loss_list
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

        # End training
        self.outputs['train']['time'] = time.time() - start_time
        self.outputs['train']['evolution'] = {'col_name': ['Epoch', 'L1_loss', 'gen_GAN_loss', 'LossG', 'LossD', 'Valid_L1'],
                                              'data': epoch_loss_list}
        logger.info(f"Finished training inpainter SN-PatchGAN in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def validate(self, dataset, save_path=None, epoch=0):
        """
        Validate the generator inpainting capabilities on a samll sample of data.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) dataset returning a small sample of fixed pairs (image, mask, idx)
            |               on which to validate the GAN over training. It must return an image tensor of dimension
            |               [C, H, W], a mask tensor of dimension [1, H, W] and a tensor of index of dimension [1].
            |               If None, no validation is performed during training.
            |---- save_path (str) path to directory where to save the inpaint results of the valida_data as .png. Each
            |               image is saved as save_path/valid_imY_epXXX.png where Y is the image index and XXX is the epoch.
            |---- epoch (int) the current epoch number.
        OUTPUT
            |---- l1_loss (float) the mean Discounted L1Loss over the validation images.
        """
        with torch.no_grad():
            # make loader
            valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                       shuffle=False, worker_init_fn=lambda _: np.random.seed())
            n_batch = len(valid_loader)
            l1_loss_fn = DiscountedL1(gamma=self.gammaL1, reduction='mean', device=self.device)

            self.generator.eval()
            # validate data by batch
            l1_loss = 0.0
            for b, data in enumerate(valid_loader):
                im_v, mask_v = data
                im_v = im_v.to(self.device).float()
                mask_v = mask_v.to(self.device).float()

                # inpaint
                im_inpaint, coarse = self.generator(im_v, mask_v)
                # recover non-masked regions
                im_inpaint = im_v * (1 - mask_v) + im_inpaint * mask_v
                coarse = im_v * (1 - mask_v) + coarse * mask_v
                # compute L1 loss
                l1_loss += l1_loss_fn(im_inpaint, im_v, mask_v).item()
                # save results
                if save_path:
                    for i in range(im_inpaint.shape[0]):
                        arr = im_inpaint[i].permute(1,2,0).squeeze().cpu().numpy()
                        io.imsave(os.path.join(save_path, f'valid_im{idx[i]}_ep{epoch}.png'), img_as_ubyte(arr))

                        arr = coarse[i].permute(1,2,0).squeeze().cpu().numpy()
                        io.imsave(os.path.join(save_path, f'valid_im{idx[i]}_coarse_ep{epoch}.png'), img_as_ubyte(arr))

                print_progessbar(b, n_batch, Name='Valid Batch', Size=40, erase=True)

        return l1_loss / n_batch

    def stabilize_BN(self, dataset, rep=5):
        """
        Perform few forward passes on discriminator and generator in train mode but without grad to stabilize batch-norm
        parameters (mean and std).
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) dataset returning a pair (image, mask) It must return an image tensor
            |               of dimension [C, H, W], a mask tensor of dimension [1, H, W].
            |---- rep (int) number of time to go over dataset.
        OUTPUT
            |---- None
        """
        with torch.no_grad():
            self.generator.train()
            self.discriminator.train()
            # make loader
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                 shuffle=False, worker_init_fn=lambda _: np.random.seed())
            n_batch = len(loader)
            # validate data by batch
            for i in range(rep):
                for b, data in enumerate(loader):
                    im, mask = data
                    im = im.to(self.device).float()
                    mask = mask.to(self.device).float()

                    # inpaint
                    _, _ = self.generator(im, mask)
                    _ = self.discriminator(im, mask)
                    # recover non-masked regions
                    print_progessbar(b, n_batch, Name=f'Stabilization Batch (iteration {i+1})', Size=40, erase=True)

    def save_models(self, export_fn, which='both'):
        """
        Save the model.
        ----------
        INPUT
            |---- export_fn (str or tuple of str) the export path(es). If a tuple if should be [path_gen, path_dis].
        OUTPUT
            |---- None
        """
        assert which in ['g', 'd', 'both'], f"Which must be one of 'g', 'd' or 'both'. Given: {which}"
        if which == 'g':
            torch.save(self.generator.state_dict(), export_fn)
        elif which == 'd':
            torch.save(self.discriminator.state_dict(), export_fn)
        elif which == 'both':
            assert isinstance(export_fn, tuple) and len(export_fn) == 2, f"With which = 'both', export_fn must be a 2-tuple."
            torch.save(self.generator.state_dict(), export_fn[0])
            torch.save(self.discriminator.state_dict(), export_fn[1])

    def load_Generator(self, import_fn, map_location='cuda'):
        """
        Load a generator model from the given path.
        ----------
        INPUT
            |---- import_fn (str) path where to get the model.
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        loaded_state_dict = torch.load(import_fn, map_location=map_location)
        self.generator.load_state_dict(loaded_state_dict)

    def load_Discriminator(self, import_fn, map_location='cuda'):
        """
        Load a discriminator model from the given path.
        ----------
        INPUT
            |---- import_fn (str) path where to get the model.
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        loaded_state_dict = torch.load(import_fn, map_location=map_location)
        self.discriminator.load_state_dict(loaded_state_dict)

    def load_models(self, import_fn, map_location='cuda'):
        """
        Load a genrator and discriminator model from the given path.
        ----------
        INPUT
            |---- import_fn (2-tuple of strings) path where to get the models (g_path, d_path).
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        self.load_Generator(import_fn[0], map_location=map_location)
        self.load_Discriminator(import_fn[1], map_location=map_location)

    def save_outputs(self, export_fn):
        """
        Save the outputs in JSON.
        ----------
        INPUT
            |---- export_fn (str) path where to get the results.
        OUTPUT
            |---- None
        """
        with open(export_fn, 'w') as fn:
            json.dump(self.outputs, fn)
