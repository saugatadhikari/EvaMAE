import argparse
import torch.nn as nn
import torch
from functools import partial
from timm.models.layers import trunc_normal_
from collections import OrderedDict

import wandb
from torch.utils.data import DataLoader

from finetuning.dataset import FloodDatasetTest, LandslideDatasetTest
from model import *
from utils import sequential_split

from metrics import Evaluator
from tqdm import tqdm
import numpy as np
import os
import h5py

import json


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

	parser.add_argument('--task', default="flood", type=str)
	parser.add_argument('--n_chans', default=3, type=int)

	parser.add_argument('--use_dem', default=0, type=int)
	parser.add_argument('--min_dem', default=None, type=float)
	parser.add_argument('--max_dem', default=None, type=float)

	parser.add_argument('--region_sizes', default=None, type=str)
	parser.add_argument('--output_path', default="./output", type=str)
	parser.add_argument('--ckpt_path', default=None, type=str)

	parser.add_argument('--use_evaloss', default=0, type=int)
	parser.add_argument('--use_evanet', default=0, type=int)
	parser.add_argument('--use_controlnet', default=0, type=int)

	return parser.parse_args()


def main():
	args = parse_args()

	with open(args.region_sizes, "r") as fp:
		region_sizes = json.load(fp)

	weight_path = args.ckpt_path
	if args.use_controlnet:
		pretrained_model = torch.load(weight_path, map_location='cpu')

		model = ControlNet(
			num_classes=args.num_classes, 
			embed_dim=1024,
			model=args.model, 
			pretrained_weight_path=weight_path,
			use_dem=args.use_dem,
			inference=True
		)

		model.load_state_dict(pretrained_model)
	
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

		model.load_state_dict(pretrained_model)
	
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

		model.load_state_dict(pretrained_model)

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

		model.load_state_dict(pretrained_model)

	elif args.model == 'evamae_crossattn':
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
		
		model.load_state_dict(pretrained_model)
		
	elif args.model == 'evamae_crossvit':
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

		model.load_state_dict(pretrained_model)

	elif args.model == 'prithvi':
		pretrained_model = torch.load(weight_path, map_location='cpu')

		model = Prithvi(
			img_size=args.img_size,
			patch_size=16,
			num_frames=1,
			tubelet_size=1,
			in_chans=3,
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

		model.load_state_dict(pretrained_model)

	elif args.model == 'satmaepp':
		pretrained_model = torch.load(weight_path, map_location='cpu')

		# Initialize the new model
		model = SatMAE(num_classes=args.num_classes,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True)
		
		model.load_state_dict(pretrained_model)

	elif args.model == 'unet':
		pretrained_model = torch.load(weight_path, map_location='cpu')
		model = UNet(args.n_chans, args.num_classes, ultrasmall=True)
		model.load_state_dict(pretrained_model)
	elif args.model == 'evanet':
		pretrained_model = torch.load(weight_path, map_location='cpu')
		model = EvaNet(args.batch_size, 3, args.num_classes, ultrasmall=True)
		model.load_state_dict(pretrained_model)
	else:
		raise ValueError('Unknown model')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = model.to(device)

	if args.task == "flood":
		args.regions = os.listdir(f"{args.base_data_path}/features/")
	elif args.task == "landslide":
		args.regions = os.listdir(f"{args.base_data_path}/img/")

	for test_region in args.regions:
		print("TEST REGION: ", test_region)
		if args.task == "flood":
			dataset = FloodDatasetTest(args.base_data_path, test_region, args.use_evaloss, args.min_dem, args.max_dem)
		elif args.task == "landslide":
			dataset = LandslideDatasetTest(args.base_data_path, test_region, args.use_evaloss, args.min_dem, args.max_dem)

		test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

		pred_patches_dict, gt_patches_dict = run_pred(model, test_loader, args, use_dem=args.use_dem, use_evaloss=args.use_evaloss, use_controlnet=args.use_controlnet)

		## Stitch image patches back together
		if args.task == "flood":
			cropped_val_test_data_path = os.path.join(args.base_data_path, "features", f"{test_region[0]}")
			cropped_val_test_data_path_gt = os.path.join(args.base_data_path, "groundTruths", f"{test_region[0]}")

			rgb_stitched, pred_stitched, gt_stitched = stitch_patches(pred_patches_dict, gt_patches_dict, cropped_val_test_data_path, f"{test_region[0]}")
		elif args.task == "landslide":
			pred_patch = pred_patches_dict[test_region]
			pred_stitched = np.transpose(pred_patch, (1, 2, 0))

			
			gt_stitched = gt_patches_dict[test_region]
			# gt_stitched = torch.tensor(gt_stitched)
			cropped_val_test_data_path = os.path.join(args.base_data_path, "TestData", "img")

			with h5py.File(os.path.join(cropped_val_test_data_path, test_region), 'r') as hf:
				image_arr = hf['img'][:]

			image_arr = np.asarray(image_arr, np.float32)
			rgb_stitched = image_arr[:,:,[1,2,3]]
		
		if args.model not in ["evanet", "unet"]: # these 2 model trained from scratch already have sigmoid applied
			sigmoid = torch.nn.Sigmoid()
			pred_sigmoid = sigmoid(torch.tensor(pred_stitched))
			pred_sigmoid = pred_sigmoid.numpy()
		else:
			pred_sigmoid = pred_stitched

		if args.task == "landslide":
			pred_stitched = pred_sigmoid[:,:,1]
			pred_stitched = np.where(pred_stitched > 0.5, 1, 0)
		else:
			if not args.use_evaloss:
				pred_stitched = pred_sigmoid[:,:,2]
				pred_stitched = np.where(pred_stitched > 0.5, 2, 0)
			else:
				pred_stitched = pred_sigmoid[:,:,0]
				pred_stitched = np.where(pred_stitched > 0.5, 1, -1)

		region_id = test_region[0].split("_")[-1]

		if args.task == "flood":
			original_height = region_sizes[region_id]["height"]
			original_width = region_sizes[region_id]["width"]
		elif args.task == "landslide":
			original_height, original_width = pred_stitched.shape[:2]

		rgb_center_cropped = center_crop(rgb_stitched, image=True, original_height=original_height, original_width=original_width)
		pred_center_cropped = center_crop(pred_stitched, image=False, original_height=original_height, original_width=original_width)
		gt_center_cropped = center_crop(gt_stitched, image=False, original_height=original_height, original_width=original_width)

		save_path = os.path.join(args.output_path, "outputs")
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		rgb_output_path = os.path.join(save_path, f"{test_region[0]}_rgb.npy")
		pred_output_path = os.path.join(save_path, f"{test_region[0]}_pred.npy")
		gt_output_path = os.path.join(save_path, f"{test_region[0]}_GT.npy")

		if args.task == "flood":
			np.save(rgb_output_path, rgb_center_cropped)
			np.save(pred_output_path, pred_center_cropped)
			np.save(gt_output_path, gt_center_cropped)

		if args.task == "landslide":
			evaluator = Evaluator(use_evaloss=args.use_evaloss, landslide=True)
		else:
			evaluator = Evaluator(use_evaloss=args.use_evaloss)

		gt_center_cropped = gt_center_cropped.astype('int')
		pred_center_cropped = pred_center_cropped.astype('int')

		flood_iou_all, dry_iou_all, mean_iou_all = evaluator.run_eval(gt_center_cropped.flatten(), pred_center_cropped.flatten())

		if args.task == "flood":
			print("----------------------------")
			print("Flood IoU: ", flood_iou_all)
			print("Dry IoU: ", dry_iou_all)
			print("Mean IoU: ", mean_iou_all)
			print("----------------------------")
		elif args.task == "landslide":
			print("----------------------------")
			print("Landslide IoU: ", flood_iou_all)
			print("Not Landslide IoU: ", dry_iou_all)
			print("Mean IoU: ", mean_iou_all)
			print("----------------------------")

def run_pred(model, data_loader, args, use_dem=False, use_evaloss=False, use_controlnet=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	evaluator = Evaluator()

	flood_iou_total = 0
	dry_iou_total = 0
	mean_iou_total = 0

	flood_ious = []
	dry_ious = []
	mean_ious = []

	total_items = 0

	epoch_correct_val = epoch_all_val = 0

	pred_patches_dict = dict()
	gt_patches_dict = dict()

	for batch in tqdm(data_loader):
		filename = batch['filename']

		data = batch["rgb_data"].cuda(non_blocking=True) # 32, 3, 224, 224
		gt_label_orig = batch["labels"]
		gt_label = batch["labels"].flatten().cuda(non_blocking=True)

		dem = batch["norm_dem_data"].cuda(non_blocking=True) # 32, 224, 224
		dem = dem.unsqueeze(1) # 32, 1, 224, 224

		if use_dem or use_controlnet:
			data = torch.cat([data, dem], dim=1)
			
		pred = model(data)

		if args.task == "landslide":
			y = pred.permute(0, 2, 3, 1).reshape(-1, 2)
		else:
			if not use_evaloss:
				y = pred.permute(0, 2, 3, 1).reshape(-1, 3)
			else:
				y = pred.permute(0, 2, 3, 1).reshape(-1, 2)

		gt_label = gt_label.detach().cpu().numpy()

		total_items += 1

		pred_labels_np = pred.detach().cpu().numpy() 
        
		## Save Image and RGB patch
		for idx in range(data.shape[0]):
			pred_patches_dict[filename[idx]] = pred_labels_np[idx, :, :, :]
			gt_patches_dict[filename[idx]] = gt_label_orig[idx, :, :]
	
	return pred_patches_dict, gt_patches_dict


def find_patch_meta(pred_patches_dict):
    y_max = 0
    x_max = 0

    for item in pred_patches_dict:

        ##print(item)

        temp = int(item.split("_")[3])
        if temp>y_max:
            y_max = temp

        temp = int(item.split("_")[5])
        if temp>x_max:
            x_max = temp


    y_max+=1
    x_max+=1
    
    return y_max, x_max


def stitch_patches(pred_patches_dict, gt_patches_dict, cropped_val_test_data_path, test_region, model=None):
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            if not model:
                dict_key = f"{test_region}_y_{i}_x_{j}_features.npy"
                rgb_patch = np.load(os.path.join(cropped_val_test_data_path, dict_key))[:, :, :3]
            elif model=="landslide":
                dict_key = test_region
                with h5py.File(os.path.join(cropped_val_test_data_path, dict_key), 'r') as hf:
                    image_arr = hf['img'][:]

                image_arr = np.asarray(image_arr, np.float32)
                rgb_patch = image_arr[:,:,[1,2,3]]
            # print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))
			
            gt_patch = gt_patches_dict[dict_key]
            # pred_patch = np.transpose(pred_patch, (1, 2, 0))

            


            if j == 0:
                rgb_x_patches = rgb_patch
                pred_x_patches = pred_patch
                gt_x_patches = gt_patch
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)
                gt_x_patches = np.concatenate((gt_x_patches, gt_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
            gt_y_patches = gt_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
            gt_y_patches = np.vstack((gt_y_patches, gt_x_patches))
	
    rgb_stitched = np.array(rgb_y_patches).astype('uint8')
    gt_stitched = np.array(gt_y_patches).astype('int')
    
    return rgb_stitched, pred_y_patches, gt_stitched


def center_crop(stictched_data, image, original_height, original_width):
    
    if image:
        current_height, current_width, _ = stictched_data.shape
    else:
        current_height, current_width = stictched_data.shape    
    
    height_diff = current_height-original_height
    width_diff = current_width-original_width
    
    cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2]
    
    return np.array(cropped)

if __name__ == "__main__":
	main()
