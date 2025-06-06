'''
This code is used to split large image into multiple patches each of size 224 x 224 for training the model. 
'''


import numpy as np
import os
import math
from glob import glob
from tqdm import tqdm


def pad_data(unpadded_data, SPATIAL_SIZE=224, is_feature = False):
    
    height = unpadded_data.shape[0]
    width = unpadded_data.shape[1]
    
#     print("height: ", height)
#     print("width: ", width)
    
    width_multiplier = math.ceil(width/SPATIAL_SIZE)
    height_multiplier = math.ceil(height/SPATIAL_SIZE)
    
#     print("width_multiplier: ", width_multiplier)
#     print("height_multiplier: ", height_multiplier)
    
    new_width = SPATIAL_SIZE*width_multiplier
    new_height = SPATIAL_SIZE*height_multiplier
#     print("new_width: ", new_width)
#     print("new_height: ", new_height)
    
    width_pad = new_width-width
    height_pad = new_height-height
    
#     print("width_pad: ", width_pad)
#     print("height_pad: ", height_pad)
    
        
    if width_pad%2 == 0:
        left = int(width_pad/2)
        right = int(width_pad/2)
    else:
        # print("Odd Width")
        left = math.floor(width_pad/2)
        right = left+1
    
    if height_pad%2 == 0:
        top = int(height_pad/2)
        bottom = int(height_pad/2)
    else:
        # print("Odd Height")
        top = math.floor(height_pad/2)
        bottom = top+1
    
#     print("left: ", left)
#     print("right: ", right)
#     print("top: ", top)
#     print("bottom: ", bottom)
        
    if is_feature:
        data_padded = np.pad(unpadded_data, pad_width = ((top, bottom),(left, right), (0, 0)), mode = 'reflect')
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(data_padded[:,:,:3].astype('int'))
    else:
        data_padded = np.pad(unpadded_data, pad_width = ((top, bottom), (left, right)), mode = 'reflect')
        
    assert data_padded.shape[0]%SPATIAL_SIZE == 0, f"Padded height must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"
    assert data_padded.shape[1]%SPATIAL_SIZE == 0, f"Padded width must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"

    # print(left, right, top, bottom)
        
#     print("data_padded: ", data_padded.shape, "\n")
    return data_padded

def crop_data(uncropped_data, output_path, TEST_REGION, is_feature = False, SPATIAL_SIZE=224):
    
    height = uncropped_data.shape[0]
    width = uncropped_data.shape[1]
    
    # print("crop input height: ", height)
    # print("crop input width: ", width)
    
    vertical_patches = height//SPATIAL_SIZE
    horizontal_patches = width//SPATIAL_SIZE
    
    # print("vertical_patches: ", vertical_patches)
    # print("horizontal_patches: ", horizontal_patches)
    # print(filename)
    
    cropped_data = []
    
    for y in range(0, vertical_patches):
        for x in range(0, horizontal_patches):
            
            if is_feature:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_features.npy"
            else:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_label.npy"
            
            # print("new_name: ", new_name)
            
            x_start = (x)*SPATIAL_SIZE
            x_end = (x+1)*SPATIAL_SIZE
            
            y_start = (y)*SPATIAL_SIZE
            y_end = (y+1)*SPATIAL_SIZE
            
            patch = uncropped_data[y_start: y_end, x_start:x_end]
            
            # print(patch.shape)
            
            np.save(os.path.join(output_path, new_name), patch)


if __name__ == "__main__":

    all_files = glob("/path/to/features/*.npy")

    for fea_file in tqdm(all_files):
        filename = os.path.basename(fea_file)
        region_id = filename.split("_")[1]

        dem_file = f"/path/to/groundTruths/Region_{region_id}_GT_Labels.npy"

        fea_arr = np.load(fea_file) # contains 4 channels, 1st 3 channels are RGB and 4th channel is DEM
        dem_arr = np.load(dem_file) # contains flood labels -> Dry: -1, Flood: 1, Unknown: 0

        padded_fea_arr = pad_data(fea_arr, is_feature=True)
        padded_dem_arr = pad_data(dem_arr, is_feature=False)

        split_folder_fea = f"/path/to/flood_patches/train/features/Region_{region_id}/" # choose train or test depending on the dataset
        split_folder_gt = f"/path/to/flood_patches/train/groundTruths/Region_{region_id}/" # choose train or test depending on the dataset
        
        os.makedirs(split_folder_fea, exist_ok=True)
        os.makedirs(split_folder_gt, exist_ok=True)

        crop_data(padded_fea_arr, split_folder_fea, TEST_REGION=region_id, is_feature=True)
        crop_data(padded_dem_arr, split_folder_gt, TEST_REGION=region_id, is_feature=False)
    