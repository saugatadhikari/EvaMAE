# EvaMAE

This repository contains the code for pre-training and fine-tuning EvaMAE models. The code for pre-training are put inside pretraining folder and the code for fine-tuning are put inside finetuning folder.  

# Instructions for running the code  

# Pre-training  
You can download the pre-training dataset here: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/Epq3McAub5ZJjedkDFK_WOEBNQ5RNsAd0n1PYmTcgfHjBQ?e=KQjMMt  

After you download the dataset and metadata files, your directory should look like:

```
<PATH_TO_DATASET_ROOT_FOLDER>  
--- train.csv  
--- val.csv  
--- EvaFlood  
------- train  
---------- flood  
--------------- flood_0    
--------------- ...  
------- val  
---------- flood    
--------------- flood_0  
--------------- ...  
```
  
For pre-training, this is the default command:  
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \  
    --nproc_per_node=4 --nnodes=1 --master_port=29201 /path/to/pretraining/main_pretrain.py \  
    --batch_size 8 \  
    --accum_iter 16 \  
    --norm_pix_loss \  
    --epochs 850 \  
    --warmup_epochs 20 \  
    --input_size 224 \  
    --patch_size 16 \  
    --mask_ratio 0.75 \  
    --model_type evamae_crossvit \  
    --dataset_type eva_flood \  
    --blr 0.000015 \  
    --num_workers 8 \  
    --spatial_mask 1 \  
    --resume /path/to/pretrained_model.pth \  
    --start_epoch 800 \  
    --train_path /path/to/train.csv \  
    --output_dir /path/to/output/folder   
```

The pre-trained models can be downloaded from here: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/Eq1vXq60xIJFo1AGWaS5R3UBqhNIOmAKMzrEZra9XdCwiA?e=Pr2dXk  

# Fine-tuning  
You can download the fine-training dataset for flood segmentation here: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/EmWszM8rUpdNsNeyKAjxmR8BbzmuHyqbGphc_6XDwLjTjQ?e=Mi1U6D  
After you download the dataset and metadata files, you should preprocess them using the code provided in ```/finetuning/data/preprocess_flood_data.py```. After preprocessing, your directory should look like:

```
<PATH_TO_DATASET_ROOT_FOLDER>  
--- train  
------- features  
----------- Region_1  
---------------- Region_1_y_0_x_0_features.npy  
---------------- ...  
----------- ...  
------- groundTruths  
----------- Region_1  
---------------- Region_1_y_0_x_0_label.npy  
---------------- ...  
----------- ...  
--- test  
------- features  
----------- Region_110  
---------------- Region_110_y_0_x_0_features.npy  
---------------- ...  
----------- ...  
------- groundTruths  
----------- Region_110  
---------------- Region_110_y_0_x_0_label.npy  
---------------- ...  
----------- ...  
```
  
For fine-tuning, this is the default command:  

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \  
--nproc_per_node=4 --nnodes=1 --master_port=1234 /path/to/main_finetune.py \  
--base_data_path="path/to/train/" \  
--model="evamae_crossvit" \  
--task="flood" \  
--num_classes=3 \  
--lr=1e-5 \  
--num_epoch=30 \  
--batch_size=32 \  
--img_size=224 \  
--use_dem=1 \  
--use_controlnet=0 \  
--min_dem=-52.147003 \  
--max_dem=143.571 \  
--output_path="/path/to/output/folder/" \  
--weight_path="/path/to/pretrained/weight.pth"  
```

The fine-tuned models can be downloaded from here: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/EseF3SNyoIFAultmmqjxmfABAXM0HvwbPmzyf3W_nxwjxw?e=uYFVTH  

