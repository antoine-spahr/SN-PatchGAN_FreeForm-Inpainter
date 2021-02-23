"""
author: Antoine Spahr

date : 17.11.2020

----------

TO DO :
"""
import sys
sys.path.append('../../')
import click
import os
import logging
import json
import random
import torch
import torch.cuda
import numpy as np
#import pandas as pd

from src.dataset.datasets import InpaintDataset, ImgMaskDataset
import src.dataset.transforms as tf
from src.models.networks.InpaintingNetwork import PatchDiscriminator, SAGatedGenerator
from src.models.optim.SNPatchGAN import SNPatchGAN

from src.utils.python_utils import AttrDict

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train an Inpainting generator with Gated convolution through a SN-PatchGAN training scheme.
    """
    # Load config file
    cfg = AttrDict.from_json_path(config_path)

    # make outputs dir
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    os.makedirs(out_path, exist_ok=True)
    if cfg.train.validate_epoch:
         os.makedirs(os.path.join(out_path, 'valid_results/'), exist_ok=True)

    # initialize seed
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # set device
    if cfg.device:
        cfg.device = torch.device(cfg.device)
    else:
        cfg.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize logger
    logger = initialize_logger(os.path.join(out_path, 'log.txt'))
    if os.path.exists(os.path.join(out_path, f'checkpoint.pt')):
        logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
    logger.info(f"Experiment : {cfg.exp_name}")

    #--------------------------------------------------------------------
    #                           Make Datasets
    #--------------------------------------------------------------------
    # load data info
    train_img_fn = glob(os.path.join(cfg.path.data_train_path, '*'))
    if cfg.dataset.n_sample >= 0:
        train_img_fn = random.sample(train_img_fn, cfg.dataset.n_sample)

    # make dataset
    train_dataset = InpaintDataset(train_img_fn,
                        augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.train.items()],
                        output_size=cfg.dataset.size, ff_param=cfg.dataset.mask)

    # load small valid subset and make dataset
    if cfg.train.validate_epoch:
        eval_img_fn = glob(os.path.join(cfg.path.data_eval_path, 'img/*'))
        eval_mask_fn = glob(os.path.join(cfg.path.data_eval_path, 'mask/*'))

        valid_dataset = ImgMaskDataset(eval_img_fn, eval_mask_fn,
                            augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.eval.items()],
                            output_size=cfg.dataset.size)
    else:
        valid_dataset = None

    logger.info(f"Train Data will be loaded from {cfg.path.data_train_path}.")
    logger.info(f"Train contains {len(train_dataset)} samples.")
    if valid_dataset:
        logger.info(f"Valid Data will be loaded from {cfg.path.data_valid_path}.")
        logger.info(f"Valid contains {len(valid_dataset)} samples.")
    logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
    if valid_dataset: logger.info(f"Evaluation online data transformation: \n\n {str(valid_dataset.transform)}\n")
    mask_params = [f"--> {k} : {v}" for k, v in cfg.dataset.mask.items()]
    logger.info("Train inpainting masks generated with \n\t" + "\n\t".join(mask_params))

    #--------------------------------------------------------------------
    #                           Make Networks
    #--------------------------------------------------------------------
    generator_net = SAGatedGenerator(**cfg.net.gen)
    discriminator_net = PatchDiscriminator(**cfg.net.dis)

    gen_params = [f"--> {k} : {v}" for k, v in cfg.net.gen.items()]
    logger.info("Gated Generator Parameters \n\t" + "\n\t".join(gen_params))
    dis_params = [f"--> {k} : {v}" for k, v in cfg.net.dis.items()]
    logger.info("Gated Discriminator Parameters \n\t" + "\n\t".join(dis_params))

    #--------------------------------------------------------------------
    #                      Make Inpainting GAN model
    #--------------------------------------------------------------------
    cfg.train.model_param.lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.train.model_param.lr_scheduler) # convert scheduler name to scheduler class object
    gan_model = SNPatchGAN(generator_net, discriminator_net, print_progress=cfg.print_progress,
                           device=cfg.device,  **cfg.train.model_param)
    train_params = [f"--> {k} : {v}" for k, v in cfg.train.model_param.items()]
    logger.info("GAN Training Parameters \n\t" + "\n\t".join(train_params))

    # load models if provided
    if cfg.train.model_path_to_load.gen:
        gan_model.load_Generator(cfg.train.model_path_to_load.gen, map_location=cfg.device)
    if cfg.train.model_path_to_load.dis:
        gan_model.load_Discriminator(cfg.train.model_path_to_load.dis, map_location=cfg.device)

    #--------------------------------------------------------------------
    #                       Train SN-PatchGAN model
    #--------------------------------------------------------------------
    if cfg.train.model_param.n_epoch > 0:
        gan_model.train(train_dataset, checkpoint_path=os.path.join(out_path, 'Checkpoint.pt'),
                        valid_dataset=valid_dataset, valid_path=os.path.join(out_path, 'valid_results/'),
                        save_freq=cfg.train.valid_save_freq)

    gan_model.stabilize_BN(train_dataset, rep=cfg.train.stabilization_rep) # perform forward pass to stabilize batchnorm parameters
    gan_model.validate(valid_dataset, os.path.join(out_path, 'valid_results/'), epoch='_final') # final validation

    #--------------------------------------------------------------------
    #                   Save outputs, models and config
    #--------------------------------------------------------------------
    # save models
    gan_model.save_models(export_fn=(os.path.join(out_path, 'generator.pt'),
                                     os.path.join(out_path, 'discriminator.pt')), which='both')
    logger.info("Generator model saved at " + os.path.join(out_path, 'generator.pt'))
    logger.info("Discriminator model saved at " + os.path.join(out_path, 'discriminator.pt'))
    # save outputs
    gan_model.save_outputs(export_fn=os.path.join(out_path, 'outputs.json'))
    logger.info("Outputs file saved at " + os.path.join(out_path, 'outputs.json'))
    # save config file
    cfg.device = str(cfg.device) # set device as string to be JSON serializable
    cfg.train.model_param.lr_scheduler = str(cfg.train.model_param.lr_scheduler)
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info("Config file saved at " + os.path.join(out_path, 'config.json'))

    # delete any checkpoints
    # if os.path.exists(os.path.join(out_path, f'Checkpoint.pt')):
    #     os.remove(os.path.join(out_path, f'Checkpoint.pt'))
    #     logger.info('Checkpoint deleted.')

def initialize_logger(logger_fn):
    """
    Initialize a logger with given file name. It will start a new logger.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    try:
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])
    except IndexError:
        pass
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logger_fn)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    main()
