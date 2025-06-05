import os
import pandas as pd
import numpy as np
import warnings
import random
import json
import cv2
from glob import glob
from typing import Any, Optional, List
import rasterio
from rasterio import logging
import copy

import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

warnings.filterwarnings('always', category=RuntimeWarning)



class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)



class EvaFlood_Dataset(SatelliteDataset):

    mean = [0.26570816290018767, 0.2921223357422886, 0.2797231069386338] # for evaflood dataset
    std = [0.14469355775119505, 0.10051454452529514, 0.08088619048723895] # for evaflood dataset

    dem_min = -7.3323298
    dem_max = 326.63733

    def __init__(self, csv_path, transform, label_type: str = 'value', categories: Optional[List[str]] = None):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param json_path: json_path (string): path to json file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=4)
        self.csv_path = csv_path
        self.base_path = '/'

        # extract base folder path from csv file path
        path_tokens = csv_path.split('/')
        for token in path_tokens:
            if '.csv' in token:
                continue

            if 'csv' in token:
                continue
            self.base_path += token.strip() + '/'

        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])
        
        self.categories = ["flood"]
        self.df = self.df[self.df['category'] == 'flood']

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform
        self.label_type = label_type
        self.data_len = len(self.df)

        assert self.data_len > 0

    def __len__(self):
        return len(self.df)

    def open_rgb_image(self, img_path, multi_channel=True):
        img = Image.open(img_path)
        img = np.array(img).astype(np.uint8)

        return img
        
    def open_tiff_image(self, img_path, multi_channel=True):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        if multi_channel:
            return img.transpose(1, 2, 0)  # (h, w, c)
        else:
            return img

    def __getitem__(self, index):
        selection = self.df.iloc[index]

        folder = 'evaflood_pretrain/train'
        if 'val' in self.csv_path:
            folder = 'evaflood_pretrain/val'

        cat = selection['category']
        loc_id = selection['location_id']
        img_id = selection['image_id']

        if cat == "flood":
            image_path = '{0}/{1}_{2}/{3}_{4}_{5}.tif'.format(cat,cat,loc_id,cat,loc_id,img_id)
            dem_path = '{0}/{1}_{2}/{3}_{4}_{5}_DEM.tif'.format(cat,cat,loc_id,cat,loc_id,img_id) # saugat
        else:
            image_path = '{0}/{1}_{2}/{3}_{4}_{5}_rgb.jpg'.format(cat,cat,loc_id,cat,loc_id,img_id)
            dem_path = '{0}/{1}_{2}/{3}_{4}_{5}_rgb_DEM.tif'.format(cat,cat,loc_id,cat,loc_id,img_id) # saugat

        abs_img_path = os.path.join(self.base_path, folder, image_path)
        abs_dem_path = os.path.join(self.base_path, folder, dem_path) # saugat

        if cat == "flood":
            images = self.open_tiff_image(abs_img_path)  # (h, w, c)
        else:
            images = self.open_rgb_image(abs_img_path)  # (h, w, c)

        # print("images_shape: ", images.shape)
        images[images < 0] = 0
        assert images.shape[2] == 3, "image shape incorrect"

        images = images.astype(np.uint8)
        
        dem = self.open_tiff_image(abs_dem_path)
        dem = dem.astype(np.float32)

        if dem.shape[:2] != images.shape[:2]:
            h,w = images.shape[:2]
            dem = cv2.resize(dem, (w, h), interpolation=cv2.INTER_LINEAR)
            dem = np.expand_dims(dem, axis=-1)

        images_w_dem = np.concatenate((images, dem), axis=-1)
        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images_w_dem)

        return {'img':img_as_tensor, 'label':labels, 'input_image_path': abs_img_path}
    
    @staticmethod
    def build_transform(is_train, input_size, mean, std, has_dem=False, dem_min=None, dem_max=None):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(ToTensorWithExtraChannel())
            t.append(NormalizeWithExtraChannel(mean, std, dem_min, dem_max))
            
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(ToTensorWithExtraChannel())
        t.append(NormalizeWithExtraChannel(mean, std, dem_min, dem_max))
        
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

class EvaFlood_Dataset_SatMAEpp(SatelliteDataset):

    mean = [0.26570816290018767, 0.2921223357422886, 0.2797231069386338] # for evaflood dataset
    std = [0.14469355775119505, 0.10051454452529514, 0.08088619048723895] # for evaflood dataset

    # overall
    dem_min = -7.3323298
    dem_max = 326.63733

    def __init__(self, csv_path, transform, label_type: str = 'value', categories: Optional[List[str]] = None):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param json_path: json_path (string): path to json file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=4)
        self.csv_path = csv_path
        self.base_path = '/'

        # extract base folder path from csv file path
        path_tokens = csv_path.split('/')
        for token in path_tokens:
            if '.csv' in token:
                continue

            if 'csv' in token:
                continue
            self.base_path += token.strip() + '/'

        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])
        
        self.categories = ["flood"]

        self.df = self.df[self.df['category'] == 'flood']

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        self.label_type = label_type
        self.data_len = len(self.df)

        assert self.data_len > 0

    def __len__(self):
        return len(self.df)

    def open_rgb_image(self, img_path, multi_channel=True):
        img = Image.open(img_path)
        img = np.array(img).astype(np.uint8)

        return img
        
    def open_tiff_image(self, img_path, multi_channel=True):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        if multi_channel:
            return img.transpose(1, 2, 0)  # (h, w, c)
        else:
            return img

    def __getitem__(self, index):
        selection = self.df.iloc[index]

        folder = 'evaflood_pretrain/train'
        if 'val' in self.csv_path:
            folder = 'evaflood_pretrain/val'

        cat = selection['category']
        loc_id = selection['location_id']
        img_id = selection['image_id']

        if cat == "flood":
            image_path = '{0}/{1}_{2}/{3}_{4}_{5}.tif'.format(cat,cat,loc_id,cat,loc_id,img_id)
            dem_path = '{0}/{1}_{2}/{3}_{4}_{5}_DEM.tif'.format(cat,cat,loc_id,cat,loc_id,img_id) # saugat
        else:
            image_path = '{0}/{1}_{2}/{3}_{4}_{5}_rgb.jpg'.format(cat,cat,loc_id,cat,loc_id,img_id)
            dem_path = '{0}/{1}_{2}/{3}_{4}_{5}_rgb_DEM.tif'.format(cat,cat,loc_id,cat,loc_id,img_id) # saugat

        abs_img_path = os.path.join(self.base_path, folder, image_path)
        abs_dem_path = os.path.join(self.base_path, folder, dem_path) # saugat

        if cat == "flood":
            images = self.open_tiff_image(abs_img_path)  # (h, w, c)
        else:
            images = self.open_rgb_image(abs_img_path)  # (h, w, c)

        images[images < 0] = 0
        assert images.shape[2] == 3, "image shape incorrect"

        images = images.astype(np.uint8)
        
        dem = self.open_tiff_image(abs_dem_path)
        dem = dem.astype(np.float32)

        if dem.shape[:2] != images.shape[:2]:
            h,w = images.shape[:2]
            dem = cv2.resize(dem, (w, h), interpolation=cv2.INTER_LINEAR)
            dem = np.expand_dims(dem, axis=-1)

        images_w_dem = np.concatenate((images, dem), axis=-1)

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images_w_dem)

        img_dn_2x = F.interpolate(img_as_tensor.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
        img_dn_4x = F.interpolate(img_dn_2x.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)

        return {'img_up_4x':img_as_tensor, 'img_up_2x':img_dn_2x, 'img':img_dn_4x, 'label':labels, 'input_image_path': abs_img_path}
    
    @staticmethod
    def build_transform(is_train, input_size, mean, std, has_dem=False, dem_min=None, dem_max=None):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:

            t.append(ToTensorWithExtraChannel())
            t.append(NormalizeWithExtraChannel(mean, std, dem_min, dem_max))
            
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(ToTensorWithExtraChannel())
        t.append(NormalizeWithExtraChannel(mean, std, dem_min, dem_max))
        
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

class NormalizeWithExtraChannel:
    def __init__(self, mean, std, dem_min, dem_max):
        self.mean = mean
        self.std = std
        self.dem_min = dem_min
        self.dem_max = dem_max

    def __call__(self, tensor):
        # tensor: shape (4, H, W)
        assert tensor.shape[0] == 4, "Expected 4 channels (RGB + extra)"

        # Apply torchvision normalization to first 3 channels
        rgb = tensor[:3, :, :]
        normalized_rgb = transforms.functional.normalize(rgb, mean=self.mean, std=self.std)

        # Min-max normalize the 4th channel
        dem = tensor[3, :, :]
        if (self.dem_max - self.dem_min) > 0:
            norm_dem = (dem - self.dem_min) / (self.dem_max - self.dem_min)
        else:
            print("Zero division error!")
            norm_dem = torch.zeros_like(dem)

        # Reassemble
        return torch.cat([normalized_rgb, norm_dem.unsqueeze(0)], dim=0)

class ToTensorWithExtraChannel:
    def __call__(self, img):
        # If input is PIL image, convert to NumPy
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Split RGB and extra channel
        rgb = img[:, :, :3]          # shape (H, W, 3)
        dem = img[:, :, 3]        # shape (H, W, 1) or (H, W)

        # Apply ToTensor to RGB
        rgb_tensor = transforms.functional.to_tensor(rgb.astype(np.uint8))  # (3, H, W), normalized to [0,1]

        # Convert extra channel to tensor (unchanged)
        dem_tensor = torch.from_numpy(extra).permute(2, 0, 1).float() if dem.ndim == 3 else torch.from_numpy(dem).unsqueeze(0).float()

        # Stack both
        return torch.cat([rgb_tensor, dem_tensor], dim=0)  # (4, H, W)
        


###################################################################################################################

def build_evaflood_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    file_path = os.path.join(args.train_path if is_train else args.test_path)

    
    if args.dataset_type == 'eva_flood': # for RGB data along with DEM
        if args.model_type == "satmaepp":
            mean = EvaFlood_Dataset_SatMAEpp.mean
            std = EvaFlood_Dataset_SatMAEpp.std

            dem_min = EvaFlood_Dataset_SatMAEpp.dem_min
            dem_max = EvaFlood_Dataset_SatMAEpp.dem_max

            transform = EvaFlood_Dataset_SatMAEpp.build_transform(is_train, args.input_size*4, mean, std, has_dem=True, dem_min=dem_min, dem_max=dem_max)
            dataset = EvaFlood_Dataset_SatMAEpp(file_path, transform)
        else:
            mean = EvaFlood_Dataset.mean
            std = EvaFlood_Dataset.std

            dem_min = EvaFlood_Dataset.dem_min
            dem_max = EvaFlood_Dataset.dem_max

            transform = EvaFlood_Dataset.build_transform(is_train, args.input_size, mean, std, has_dem=True, dem_min=dem_min, dem_max=dem_max)
            dataset = EvaFlood_Dataset(file_path, transform)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    return dataset
