# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import math
import time
from pathlib import Path
import cv2
#import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.datasets import build_evaflood_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import evamae_channel_predict, evamae_conv_predict, evamae_crossvit_predict # our models


def get_args_parser():
    parser = argparse.ArgumentParser('SatMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='group_c', choices=['evamae_channel', 'evamae_crossvit', 'evamae_conv'], help='Use channel model')
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=96, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--spatial_mask', type=int, default=0,
                        help='Whether to mask all channels of a spatial location. Only for indp c model')
    parser.add_argument('--conv_kernel_size', type=int, default=3,
                        help='Size of conv kernel for dem encoding')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.0001, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default='dataset/evaflood_pretrain/train.csv', type=str, help='Train .csv path')
    parser.add_argument('--test_path', default='dataset/evaflood_pretrain/test.csv', type=str, help='Train .csv path')
    parser.add_argument('--dataset_type', default='sentinel', choices=['rgb', 'eva_flood'],
                        help='Whether to use rgb, or eva_flood dataset.')
    parser.add_argument('--masked_bands', type=int, nargs='+', default=None,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC mae")

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--output_image_folder_name', default='/images', help='folder where to save image')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint; path to saved checkpoint')
    parser.add_argument('--wandb', type=str, default=None,help="Wandb project name, eg: sentinel_pretrain")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)  # prev default was -1
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--get_all_channels_loss', type=int, default=0,
                        help='For eval, whether to get loss from all channel or just RGB channels')

    return parser

def unpatchify(x, p, c):
    """
    x: (N, L, C*patch_size**2)
    p: Patch embed patch size
    c: Number of channels
    imgs: (N, C, H, W)
    """
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
    x = torch.einsum('nhwcpq->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = build_evaflood_dataset(is_train=False, args=args)
    # print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.model_type == 'evamae_channel':
        model = evamae_channel_predict.__dict__[args.model](img_size=args.input_size,
                                                patch_size=args.patch_size,
                                                in_chans=dataset_test.in_c,
                                                norm_pix_loss=args.norm_pix_loss)
    elif args.model_type == 'evamae_crossvit':
        dataset_test.in_c = 3
        model = evamae_crossvit_predict.__dict__[args.model](img_size=args.input_size,
                                                                patch_size=args.patch_size,
                                                                in_chans=dataset_test.in_c,
                                                                norm_pix_loss=args.norm_pix_loss)                                                            
    elif args.model_type == 'evamae_conv':
        dataset_test.in_c = 3
        model = evamae_conv_predict.__dict__[args.model](img_size=args.input_size,
                                                                patch_size=args.patch_size,
                                                                in_chans=dataset_test.in_c,
                                                                norm_pix_loss=args.norm_pix_loss,
                                                                conv_kernel_size=args.conv_kernel_size)
    
    
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    '''
    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)
    '''

    start_time = time.time()
    print("start_epoch: ", args.start_epoch)

    if args.distributed:
        data_loader_test.sampler.set_epoch(args.start_epoch)

    model.eval()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    images_out_path = args.output_dir + args.output_image_folder_name

    if not os.path.exists(images_out_path):
        os.mkdir(images_out_path)

    counter = 0
    for data_iter_step, samples in enumerate(data_loader_test):
        # if counter > 5:
        #     break

        images, input_image_path = samples['img'], samples['input_image_path'][0]
        filename = os.path.basename(input_image_path).split(".")[0]

        images = images.to(device, non_blocking=True)
        loss, pred, mask, mean_, var_ = model(images, mask_ratio=args.mask_ratio, train=args.get_all_channels_loss, args=args, images_out_path=images_out_path, filename=filename)
        
        print(f"filename: {filename}       MSE_loss: {loss}")

        counter += 1



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
