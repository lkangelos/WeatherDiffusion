import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def overlapping_grid_indices(x_cond, output_size, r=None):
    _, c, h, w = x_cond.shape
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    return h_list, w_list

def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    # model.restore(val_loader, validation=args.test_set, r=args.grid_r)

    image_folder = os.path.join(args.image_folder, config.data.dataset, args.test_set)
    p_size = config.data.image_size
    h_list, w_list = overlapping_grid_indices(x_cond, output_size=p_size, r=args.grid_r)
    corners = [(i, j) for i in h_list for j in w_list]
    x = torch.randn(x_cond.size(), device=diffusion.device)
    x_output = diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
    x_output = inverse_data_transform(x_output)
    utils.logging.save_image(x_output, os.path.join(image_folder, "output.png"))


if __name__ == '__main__':
    main()
