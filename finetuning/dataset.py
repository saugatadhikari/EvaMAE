import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import re
import h5py
import cv2


class FloodDataset(Dataset):
	def __init__(self, base_data_path, use_evaloss=False, min_dem=None, max_dem=None, size=None, start=0):
		self.base_data_path = base_data_path
		self.use_evaloss = use_evaloss
		self.min_dem = min_dem
		self.max_dem = max_dem
		self.imgs = []
		self.labels = []
		self.dems = []

		self.regions = os.listdir(f"{self.base_data_path}/features/")

		for t_r in self.regions:
			imgs = os.listdir(f"{self.base_data_path}/features/{t_r}")
			self.imgs.extend(imgs)

		training_transforms = []

		# considering all regions
		training_transforms += [transforms.Normalize(mean=[77.14367257949118, 76.34090872507566, 71.48507427177988],
													std=[39.56007031111885, 30.198102940459442, 26.456583540575238])]

		data_transforms = transforms.Compose(training_transforms)
		self.data_transforms = data_transforms

	def __len__(self):
		"""
		Returns:
			int: The size of the dataset.
		"""
		return len(self.imgs)

	def __getitem__(self, img_idx):
		self.data_dict = dict()

		fea_file = self.imgs[img_idx] # Region_20_y_0_x_0_features.npy

		label_file = re.sub("features", "label", fea_file)

		region_id = int(fea_file.split("_")[1])

		# load features
		feature_path = os.path.join(self.base_data_path, "features", f"Region_{region_id}", fea_file)
		features_arr = np.load(feature_path)

		# load GT labels
		label_path = os.path.join(self.base_data_path, "groundTruths", f"Region_{region_id}", label_file)
		label_arr = np.load(label_path)
		label_arr = label_arr.astype('long')

		# separate rgb and dem arr
		# rgb_arr = features_arr[:,:,:3].astype('uint8')
		rgb_arr = features_arr[:,:,:3]
		dem_arr = features_arr[:,:,3].astype('float32')

		if not self.use_evaloss: # need -1, 0, 1 for evaloss, 0, 1, 2 for CrossEntropy loss
			label_arr = label_arr + 1
		
		rgb_arr = torch.tensor(rgb_arr.astype('float32')).permute(2, 0, 1)
		self.transformed_rgb = self.data_transforms(rgb_arr)
		self.norm_dem = self.normalize(dem_arr)

		# put all data in one dictionary
		self.data_dict["filename"] = fea_file
		self.data_dict['rgb_data'] = self.transformed_rgb
		self.data_dict['dem_data'] = dem_arr
		self.data_dict['norm_dem_data'] = self.norm_dem
		self.data_dict['labels'] = label_arr

		return self.data_dict
	
	def normalize(self, data):
        
		normalized_data = (data - (self.min_dem))/(self.max_dem - self.min_dem)
        
		assert np.min(normalized_data) >=0, "Normalized value should be greater than equal 0"
		assert np.max(normalized_data) <=1, "Normalized value should be lesser than equal 1"
        
		return normalized_data

	
class FloodDatasetTest(Dataset):
	def __init__(self, base_data_path, regions, use_evaloss=False, min_dem=None, max_dem=None, size=None, start=0):
		self.base_data_path = base_data_path
		self.regions = regions
		self.use_evaloss = use_evaloss
		self.min_dem = min_dem
		self.max_dem = max_dem
		self.imgs = []
		self.labels = []
		self.dems = []

		# for t_r in self.regions:
		imgs = os.listdir(f"{self.base_data_path}/features/{self.regions[0]}")

		self.imgs.extend(imgs)
		
		# transform using EvaNet's methods
		training_transforms = []

		# considering all regions
		training_transforms += [transforms.Normalize(mean=[77.14367257949118, 76.34090872507566, 71.48507427177988],
													std=[39.56007031111885, 30.198102940459442, 26.456583540575238])]

		data_transforms = transforms.Compose(training_transforms)
		self.data_transforms = data_transforms

	def __len__(self):
		"""
		Returns:
			int: The size of the dataset.
		"""
		return len(self.imgs)

	def __getitem__(self, img_idx):
		self.data_dict = dict()

		fea_file = self.imgs[img_idx] # Region_20_y_0_x_0_features.npy

		label_file = re.sub("features", "label", fea_file)

		region_id = int(fea_file.split("_")[1])

		# load features
		feature_path = os.path.join(self.base_data_path, "features", f"Region_{region_id}", fea_file)
		features_arr = np.load(feature_path)

		# load GT labels
		label_path = os.path.join(self.base_data_path, "groundTruths", f"Region_{region_id}", label_file)
		label_arr = np.load(label_path)
		label_arr = label_arr.astype('long')

		rgb_arr = features_arr[:,:,:3]
		dem_arr = features_arr[:,:,3].astype('float32')

		if not self.use_evaloss: # need -1, 0, 1 for evaloss, 0, 1, 2 for CrossEntropy loss
			label_arr = label_arr + 1
		
		# Apply torchvision tranforms to rgb data (conver to tensor and normalize using mean std of 0.5 !!!!!!!!!!)
		rgb_arr = torch.tensor(rgb_arr.astype('float32')).permute(2, 0, 1)
		
		self.transformed_rgb = self.data_transforms(rgb_arr)
		self.norm_dem = self.normalize(dem_arr)

		# put all data in one dictionary
		self.data_dict["filename"] = fea_file
		self.data_dict['rgb_data'] = self.transformed_rgb
		self.data_dict['dem_data'] = dem_arr
		self.data_dict['norm_dem_data'] = self.norm_dem
		self.data_dict['labels'] = label_arr

		return self.data_dict
	
	def normalize(self, data):
        
		normalized_data = (data - (self.min_dem))/(self.max_dem - self.min_dem)
        
		assert np.min(normalized_data) >=0, "Normalized value should be greater than equal 0"
		assert np.max(normalized_data) <=1, "Normalized value should be lesser than equal 1"
        
		return normalized_data

class LandslideDataset(Dataset):
	def __init__(self, base_data_path, regions, use_evaloss=False, min_dem=None, max_dem=None, size=None, start=0):
		self.base_data_path = base_data_path
		self.regions = regions
		self.use_evaloss = use_evaloss
		self.min_dem = min_dem
		self.max_dem = max_dem
		self.imgs = []
		self.labels = []
		self.dems = []

		self.counter = 0

		# all channels in Landslide4Sense
		self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
		self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

		self.imgs = os.listdir(f"{self.base_data_path}/img/")

	def __len__(self):
		"""
		Returns:
			int: The size of the dataset.
		"""
		return len(self.imgs)

	def __getitem__(self, img_idx):
		self.data_dict = dict()

		fea_file = self.imgs[img_idx]

		label_file = re.sub("image", "mask", fea_file)

		feature_path = os.path.join(self.base_data_path, "img", fea_file)
		label_path = os.path.join(self.base_data_path, "mask", label_file)

		with h5py.File(feature_path, 'r') as hf:
			image = hf['img'][:]

		with h5py.File(label_path, 'r') as hf:
			label = hf['mask'][:]
		
		image = np.asarray(image, np.float32)
		label = np.asarray(label, np.float32)
		# label = label.long()
		image = image.transpose((-1, 0, 1))
		size = image.shape

		dem_orig = image[13, :, :]

		for i in range(len(self.mean)):
			image[i,:,:] -= self.mean[i]
			image[i,:,:] /= self.std[i]

		# only get 4 channels, RGB and Elevation
		dem = image[13, :, :]
		image = image[[1,2,3], :, :]
		size = image.shape

		image_rolled = image.transpose(1,2,0)
		image_upsampled = cv2.resize(image_rolled, (224, 224), interpolation=cv2.INTER_CUBIC)
		image_upsampled = image_upsampled.transpose(2, 0, 1)

		dem_upsampled = cv2.resize(dem, (224, 224), interpolation=cv2.INTER_CUBIC)
		dem_orig_upsampled = cv2.resize(dem_orig, (224, 224), interpolation=cv2.INTER_CUBIC)

		label_upsampled = cv2.resize(label, (224, 224), interpolation=cv2.INTER_CUBIC)

		self.data_dict["filename"] = fea_file
		self.data_dict['rgb_data'] = image_upsampled
		self.data_dict['dem_data'] = dem_orig_upsampled
		self.data_dict['norm_dem_data'] = dem_upsampled
		self.data_dict['labels'] = label_upsampled


		return self.data_dict
	
	def normalize(self, data):
        
		normalized_data = (data - (self.min_dem))/(self.max_dem - self.min_dem)
        
		assert np.min(normalized_data) >=0, "Normalized value should be greater than equal 0"
		assert np.max(normalized_data) <=1, "Normalized value should be lesser than equal 1"
        
		return normalized_data

class LandslideDatasetTest(Dataset):
	def __init__(self, base_data_path, regions, use_evaloss=False, min_dem=None, max_dem=None, size=None, start=0):
		self.base_data_path = base_data_path
		self.regions = regions
		self.use_evaloss = use_evaloss
		self.min_dem = min_dem
		self.max_dem = max_dem
		self.imgs = []
		self.labels = []
		self.dems = []

		self.counter = 0

		self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
		self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

		self.imgs = [self.regions]

	def __len__(self):
		"""
		Returns:
			int: The size of the dataset.
		"""
		return len(self.imgs)

	def __getitem__(self, img_idx):
		self.data_dict = dict()

		fea_file = self.imgs[img_idx] # Region_20_y_0_x_0_features.npy

		label_file = re.sub("image", "mask", fea_file)

		feature_path = os.path.join(self.base_data_path, "img", fea_file)
		label_path = os.path.join(self.base_data_path, "mask", label_file)

		with h5py.File(feature_path, 'r') as hf:
			image = hf['img'][:]

		with h5py.File(label_path, 'r') as hf:
			label = hf['mask'][:]
		
		image = np.asarray(image, np.float32)
		label = np.asarray(label, np.float32)
		# label = label.long()
		image = image.transpose((-1, 0, 1))
		size = image.shape

		dem_orig = image[13, :, :]

		for i in range(len(self.mean)):
			image[i,:,:] -= self.mean[i]
			image[i,:,:] /= self.std[i]

		# only get 4 channels, RGB and Elevation
		dem = image[13, :, :]
		image = image[[1,2,3], :, :]
		size = image.shape

		image_rolled = image.transpose(1,2,0)
		image_upsampled = cv2.resize(image_rolled, (224, 224), interpolation=cv2.INTER_CUBIC)
		image_upsampled = image_upsampled.transpose(2, 0, 1)

		dem_upsampled = cv2.resize(dem, (224, 224), interpolation=cv2.INTER_CUBIC)
		dem_orig_upsampled = cv2.resize(dem_orig, (224, 224), interpolation=cv2.INTER_CUBIC)

		label_upsampled = cv2.resize(label, (224, 224), interpolation=cv2.INTER_CUBIC)

		self.data_dict["filename"] = fea_file
		self.data_dict['rgb_data'] = image_upsampled
		self.data_dict['dem_data'] = dem_orig_upsampled
		self.data_dict['norm_dem_data'] = dem_upsampled
		self.data_dict['labels'] = label_upsampled

		return self.data_dict
	
	def normalize(self, data):
        
		normalized_data = (data - (self.min_dem))/(self.max_dem - self.min_dem)
        
		assert np.min(normalized_data) >=0, "Normalized value should be greater than equal 0"
		assert np.max(normalized_data) <=1, "Normalized value should be lesser than equal 1"
        
		return normalized_data