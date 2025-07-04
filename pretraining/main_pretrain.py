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
import time
from pathlib import Path
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

import models_mae, prithvi, satmaepp # baseline models
import evamae_channel, evamae_conv, evamae_conv_minus, evamae_crossattn, evamae_crossvit # our models

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('EvaMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='evamae_channel', choices=['evamae_channel', 'evamae_conv', 'evamae_conv_minus', 'evamae_crossattn', 'evamae_crossvit', 'prithvi', 'satmaepp'], help='Use channel model')
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
    parser.add_argument('--output_image_folder_name', default='/images', help='folder where to save image')

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
    parser.add_argument('--dataset_type', default='eva_flood', choices=['rgb', 'eva_flood'],
                        help='Whether to use rgb, or eva_flood dataset.')
    parser.add_argument('--masked_bands', type=int, nargs='+', default=None,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC mae")

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
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

    parser.add_argument('--additional_channel', type=int, default=0,
                        help='Whether the new model has extra (DEM) channel or not. Only required for evamae_channel')

    return parser


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

    dataset_train = build_evaflood_dataset(is_train=True, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.model_type == 'evamae_channel':
        model = evamae_channel.__dict__[args.model](img_size=args.input_size,
                                                patch_size=args.patch_size,
                                                in_chans=dataset_train.in_c,
                                                norm_pix_loss=args.norm_pix_loss)
        if args.additional_channel:
            model_satmae = models_mae.__dict__[args.model](img_size=args.input_size,
                                                patch_size=args.patch_size,
                                                in_chans=dataset_train.in_c,
                                                norm_pix_loss=args.norm_pix_loss)
    elif args.model_type == 'evamae_conv':
        dataset_train.in_c = 3
        model = evamae_conv.__dict__[args.model](img_size=args.input_size,
                                                                patch_size=args.patch_size,
                                                                in_chans=dataset_train.in_c,
                                                                norm_pix_loss=args.norm_pix_loss,
                                                                conv_kernel_size=args.conv_kernel_size)
    elif args.model_type == 'evamae_conv_minus':
        dataset_train.in_c = 3
        model = evamae_conv_minus.__dict__[args.model](img_size=args.input_size,
                                                                patch_size=args.patch_size,
                                                                in_chans=dataset_train.in_c,
                                                                norm_pix_loss=args.norm_pix_loss,
                                                                conv_kernel_size=args.conv_kernel_size)     
    elif args.model_type == 'evamae_crossattn':
        dataset_train.in_c = 3
        model = evamae_crossattn.__dict__[args.model](img_size=args.input_size,
                                                            patch_size=args.patch_size,
                                                            in_chans=dataset_train.in_c,
                                                            norm_pix_loss=args.norm_pix_loss)  
    elif args.model_type == 'evamae_crossvit':
        dataset_train.in_c = 3
        model = evamae_crossvit.__dict__[args.model](img_size=args.input_size,
                                                            patch_size=args.patch_size,
                                                            in_chans=dataset_train.in_c,
                                                            norm_pix_loss=args.norm_pix_loss)                                                                                             
    elif args.model_type == 'prithvi':
        dataset_train.in_c = 3
        model = prithvi.__dict__[args.model](img_size=args.input_size,
                                                                patch_size=args.patch_size,
                                                                in_chans=dataset_train.in_c,
                                                                norm_pix_loss=args.norm_pix_loss)
    elif args.model_type == 'satmaepp':
        print("SatMAEpp")
        dataset_train.in_c = 3
        model = satmaepp.__dict__[args.model](img_size=args.input_size,
                                                                patch_size=args.patch_size,
                                                                in_chans=dataset_train.in_c,
                                                                norm_pix_loss=args.norm_pix_loss)                                                            
    else:
        model = models_mae.__dict__[args.model](img_size=args.input_size,
                                                patch_size=args.patch_size,
                                                in_chans=dataset_train.in_c,
                                                norm_pix_loss=args.norm_pix_loss)
    
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    if args.additional_channel:
        misc.load_model_resume(args=args, new_model=model_without_ddp, old_model=model_satmae, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    '''
    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)
    '''

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    print("start_epoch: ", args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        model.train() 

        print("EPOCH: ", epoch)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
            
        # if args.output_dir:
        if args.output_dir and ((epoch+1) % 2 == 0 or (epoch + 1) == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            '''
            try:
                wandb.log(log_stats)
            except ValueError:
                print(f"Invalid stats?")
            '''

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
