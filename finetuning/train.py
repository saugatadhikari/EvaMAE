import os
import torch
from tqdm import tqdm
import numpy as np
import wandb

from utils import EarlyStopping


def train(args, model, train_loader, val_loader, optimizer, criterion, scheduler, num_epoch, early_stop_patience, output_path, use_evaloss=False, use_dem=False):
	early_stop = EarlyStopping(early_stop_patience)
	grad_scaler = torch.cuda.amp.GradScaler()

	sigmoid = torch.nn.Sigmoid()

	best_val = -100
	model = model.cuda()
	for epoch in range(args.resume_epoch, num_epoch):
		print(f"Epoch {epoch}")

		if args.distributed:
			train_loader.sampler.set_epoch(epoch)

		model.train()
		batch_losses = []
		epoch_correct = epoch_all = 0
		for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epoch}'):

			data = batch["rgb_data"].cuda(non_blocking=True) # 32, 3, 224, 224
			label = batch["labels"].flatten().cuda(non_blocking=True)
			dem_norm = batch["norm_dem_data"].cuda(non_blocking=True) # 32, 224, 224
			dem_norm = dem_norm.unsqueeze(1) # 32, 1, 224, 224

			dem_orig = batch["dem_data"].cuda(non_blocking=True) # 32, 224, 224

			if use_dem or args.use_controlnet:
				data = torch.cat([data, dem_norm], dim=1)

			with torch.cuda.amp.autocast():
				pred = model(data).permute(0, 2, 3, 1)   
				
				if args.model not in ["evanet", "unet"]:
					pred = sigmoid(pred)

				if args.use_evaloss:
					y = pred.reshape(-1, 2)

					label_orig = batch["labels"].cuda(non_blocking=True)
					dem_orig = dem_orig.unsqueeze(1).float()
					label_orig = label_orig.unsqueeze(1).float()
					pred_permuted = pred.permute(0, 3, 1, 2).float()  
					loss = criterion.forward(pred_permuted, dem_orig, label_orig)
				else:
					y = pred.reshape(-1, args.num_classes)
					loss = criterion(y, label)

			optimizer.zero_grad()
			grad_scaler.scale(loss).backward()
			grad_scaler.step(optimizer)
			grad_scaler.update()

			torch.cuda.synchronize()

			batch_losses.append(loss.item())

			y = sigmoid(y)

			output_label = torch.argmax(y, dim=-1)
			if args.use_evaloss:
				mask = label != 0
			else:
				mask = label != 1
			epoch_correct += (label[mask] == output_label[mask]).sum().item()
			epoch_all += label[mask].shape[0]

		epoch_loss = np.mean(batch_losses)
		acc = epoch_correct / epoch_all

		scheduler.step(epoch_loss)

		# ----------------Validation----------------
		model.eval()
		with torch.no_grad():
			batch_losses_val = []
			epoch_correct_val = epoch_all_val = 0
			for batch in val_loader:

				data = batch["rgb_data"].cuda(non_blocking=True)
				label = batch["labels"].flatten().cuda(non_blocking=True)

				dem_norm = batch["norm_dem_data"].cuda(non_blocking=True) # 32, 224, 224
				dem_norm = dem_norm.unsqueeze(1) # 32, 1, 224, 224

				dem_orig = batch["dem_data"].cuda(non_blocking=True) # 32, 224, 224

				if use_dem:
					data = torch.cat([data, dem_norm], dim=1)
					
				y = model(data).permute(0, 2, 3, 1).reshape(-1, 3)

				if use_evaloss:
					loss = criterion.forward(y, dem_orig, label)
				else:
					loss = criterion(y, label)
					
				batch_losses_val.append(loss.item())

				output_label = torch.argmax(y, dim=-1)
				if use_evaloss:
					mask = label != 0
				else:
					mask = label != 1
				epoch_correct_val += (label[mask] == output_label[mask]).sum().item()
				epoch_all_val += label[mask].shape[0]

		epoch_loss_val = np.mean(batch_losses_val)
		acc_val = epoch_correct_val / epoch_all_val

		if acc_val > best_val:
			best_val = acc_val
			torch.save(model.state_dict(), f"{output_path}/model_epoch_{epoch}.pth")

		early_stop(epoch_loss_val)
		if early_stop.early_stop:
			print("Early Stop!")
			break
