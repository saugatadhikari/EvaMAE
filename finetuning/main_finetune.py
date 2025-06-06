import os
import argparse
import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F

from torch.utils.data import DataLoader

from finetuning.dataset import FloodDataset, LandslideDataset
from model import *
from train import train
from utils import sequential_split

import util.misc as misc
from loss import ElevationLoss

import numpy as np

import time

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_data_path', default=None, type=str)
	parser.add_argument('--regions', nargs='+', type=str, help='all regions train/test', action='append', default=[])
	parser.add_argument('--model', default=None,
						choices=['satmae', 'evamae_channel', 'evamae_conv', 'evamae_crossattn', 'evamae_crossvit', 'satmaepp', 'prithvi', 'unet', 'evanet'], type=str)

	# training parameters
	parser.add_argument('--num_classes', default=3, type=int)
	parser.add_argument('--num_epoch', default=50, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--sche_pat', default=3, type=int)
	parser.add_argument('--early_stop_pat', default=10, type=int)

	# models parameters
	parser.add_argument('--img_size', default=224, type=int)

	parser.add_argument('--use_dem', default=0, type=int)
	parser.add_argument('--min_dem', default=None, type=float)
	parser.add_argument('--max_dem', default=None, type=float)

	parser.add_argument('--output_path', default="./output", type=str)
	parser.add_argument('--weight_path', default=None, type=str)

	parser.add_argument('--task', default="flood", type=str)

	parser.add_argument('--use_evaloss', default=0, type=int)
	parser.add_argument('--use_evanet', default=0, type=int)

	parser.add_argument('--use_controlnet', default=0, type=int)
	parser.add_argument('--n_chans', default=3, type=int)

	parser.add_argument('--resume_epoch', default=0, type=int)

	parser.add_argument('--device', default='cuda', help='device to use for training / testing')
	parser.add_argument('--seed', default=0, type=int)

	parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
	parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)  # prev default was -1
	parser.add_argument('--dist_on_itp', action='store_true')
	parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

	return parser.parse_args()


def main():
	args = parse_args()

	misc.init_distributed_mode(args)

	print("use_dem: ", args.use_dem)

	device = torch.device(args.device)

	# fix the seed for reproducibility
	seed = args.seed + misc.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)


	if args.task == "flood":
		dataset = FloodDataset(args.base_data_path, args.use_evaloss, args.min_dem, args.max_dem)
	elif args.task == "landslide":
		dataset = LandslideDataset(args.base_data_path, args.use_evaloss, args.min_dem, args.max_dem)

	train_set, val_set = sequential_split(dataset, train_ratio=0.8)

	if True:  # args.distributed:
		num_tasks = misc.get_world_size()
		global_rank = misc.get_rank()
		sampler_train = torch.utils.data.DistributedSampler(
			train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
		)
		sampler_val = torch.utils.data.DistributedSampler(
			val_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
		)
		print("Sampler_train = %s" % str(sampler_train))
		print("Sampler_val = %s" % str(sampler_val))
	else:
		sampler_train = torch.utils.data.RandomSampler(train_set)
		sampler_val = torch.utils.data.RandomSampler(val_set)

	train_loader = DataLoader(train_set, sampler=sampler_train, batch_size=args.batch_size, pin_memory=True)
	val_loader = DataLoader(val_set, sampler=sampler_val, batch_size=args.batch_size, pin_memory=True, drop_last=True)

	weight_path = args.weight_path

	if args.use_controlnet:
		model = ControlNet(
			num_classes=args.num_classes, 
			embed_dim=1024,
			model=args.model, 
			pretrained_weight_path=weight_path,
			use_dem=args.use_dem
		)

		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

	elif args.model == 'satmae':
		pretrained_model = torch.load(weight_path, map_location='cpu')['model']

		# Initialize the new model
		model = SatMAE(num_classes=args.num_classes,
				 		img_size=args.img_size,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True)
		
		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		model_dict = model_without_ddp.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}

		model_dict.update(pretrained_dict)
		model_without_ddp.load_state_dict(model_dict)
	
	elif args.model == 'evamae_channel':
		pretrained_model = torch.load(weight_path, map_location='cpu')['model']

		# Initialize the new model
		model = EvaMAE_Channel(num_classes=args.num_classes,
						img_size=args.img_size,
						in_chans=4,
						patch_size=16,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True)
		
		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		# Load weights, ignoring the head
		model_dict = model_without_ddp.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}
		model_dict.update(pretrained_dict)
		model_without_ddp.load_state_dict(model_dict)

	elif args.model == 'evamae_conv':
		pretrained_model = torch.load(weight_path, map_location='cpu')['model']

		# Initialize the new model
		model = EvaMAE_Conv(num_classes=args.num_classes,
						img_size=args.img_size,
						in_chans=3,
						patch_size=16,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True,
					   conv_kernel_size=3,
					   use_double_conv=True)
		
		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		# Load weights, ignoring the head
		model_dict = model_without_ddp.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}
		model_dict.update(pretrained_dict)
		model_without_ddp.load_state_dict(model_dict)

	elif args.model == 'evamae_crossattn':
		if args.resume_epoch == 0:
			pretrained_model = torch.load(weight_path, map_location='cpu')['model']
		else:
			pretrained_model = torch.load(weight_path, map_location='cpu')

		# Initialize the new model
		model = EvaMAE_CrossAttn(num_classes=args.num_classes,
						img_size=args.img_size,
						in_chans=3,
						patch_size=16,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True,
					   use_evanet=args.use_evanet)

		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		# Load weights, ignoring the head
		model_dict = model_without_ddp.state_dict()

		if args.resume_epoch == 0:
			pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}
			model_dict.update(pretrained_dict)
			model_without_ddp.load_state_dict(model_dict)
		else:
			model_without_ddp.load_state_dict(pretrained_model)
		
	elif args.model == 'evamae_crossvit':
		if args.resume_epoch == 0:
			pretrained_model = torch.load(weight_path, map_location='cpu')['model']
		else:
			pretrained_model = torch.load(weight_path, map_location='cpu')

		# Initialize the new model
		model = EvaMAE_CrossViT(num_classes=args.num_classes,
						img_size=args.img_size,
						in_chans=3,
						patch_size=16,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True,
					   use_evanet=args.use_evanet)

		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		# Load weights, ignoring the head
		model_dict = model_without_ddp.state_dict()
		if args.resume_epoch == 0:
			pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}
			model_dict.update(pretrained_dict)
			model_without_ddp.load_state_dict(model_dict)
		else:
			model_without_ddp.load_state_dict(pretrained_model)	

	elif args.model == 'prithvi':
		pretrained_model = torch.load(weight_path, map_location='cpu')

		model = Prithvi(
			img_size=args.img_size,
			patch_size=16,
			num_frames=1,
			tubelet_size=1,
			in_chans=6,
			embed_dim=768,
			depth=12,
			num_heads=12,
			decoder_embed_dim=512,
			decoder_depth=8,
			decoder_num_heads=16,
			mlp_ratio=4.0,
			norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
			norm_pix_loss=False,
		)

		model.to(device)
		model_without_ddp = model

		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		model_without_ddp.load_state_dict(pretrained_model, strict=False)

		old_proj = model_without_ddp.patch_embed.proj
		model_without_ddp.patch_embed = PatchEmbedPrithvi(args.img_size, 16, 3, 1, 3, 768)
		model_without_ddp.patch_embed.proj.weight = nn.Parameter(old_proj.weight[:, :3, :, :, :].clone()) # only take RGB

		model_without_ddp = PrithviSeg(args.num_classes, 768, model_without_ddp)

	elif args.model == 'satmaepp':
		pretrained_model = torch.load(weight_path, map_location='cpu')

		# Initialize the new model
		model = SatMAE(num_classes=args.num_classes,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True)
		
		model.to(device)
		model_without_ddp = model
		
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		model_dict = model_without_ddp.state_dict()

		if args.resume_epoch == 0:
			pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}
			model_dict.update(pretrained_dict)
			model_without_ddp.load_state_dict(model_dict)
		else:
			model_without_ddp.load_state_dict(pretrained_model)

	elif args.model == 'unet':
		model = UNet(args.n_chans, args.num_classes, ultrasmall=True)

		model = model.to(device)

		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module
	elif args.model == 'evanet':
		model = EvaNet(args.batch_size, 3, args.num_classes, ultrasmall=True)
		model = model.to(device)

		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module
	else:
		raise ValueError('Unknown model')

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

	if args.use_evaloss:
		criterion = ElevationLoss()
	else:
		if args.task == "landslide":
			criterion = torch.nn.CrossEntropyLoss()
		else:
			criterion = torch.nn.CrossEntropyLoss(ignore_index=1) # original label is [-1, 0, 1], 0 is unknown, +1 to set to positive
		
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.sche_pat, factor=0.1, threshold=1e-4)

	start = time.time()
	train(args, model_without_ddp, train_loader, val_loader, optimizer, criterion, scheduler, args.num_epoch, args.early_stop_pat, args.output_path, use_dem=args.use_dem)
	end = time.time()

	total_time = end - start
	print("Training time: ", total_time)
	

if __name__ == "__main__":

	main()
