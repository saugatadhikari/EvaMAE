import torch
from torch.utils.data import Subset
import random


def sequential_split(dataset, train_ratio=0.8):
	# Calculate the number of samples to include in each set
	train_size = int(len(dataset) * train_ratio)

	# Generate indices: here, it's simple since it's just a range
	indices = list(range(len(dataset)))

	# Split indices into training and validation parts
	train_indices = indices[:train_size]
	val_indices = indices[train_size:]

	# Create PyTorch subsets
	train_subset = Subset(dataset, train_indices)
	val_subset = Subset(dataset, val_indices)

	return train_subset, val_subset



class EarlyStopping:
	"""
	Early stopping to stop the training when the loss does not improve after certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is not improving
		:param min_delta:
		minimum difference between new loss and old loss for new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			# reset counter if validation loss improves
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			# logging.info(f"Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				# logging.info('Early stop!')
				self.early_stop = True
