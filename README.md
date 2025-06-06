# EvaMAE

This repository contains the code for pre-training and fine-tuning EvaMAE models. The code for pre-training are put inside pretraining folder and the code for fine-tuning are put inside finetuning folder.  

# Instructions for running the code  

# Pre-training  
You can download the pre-training dataset here: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/Epq3McAub5ZJjedkDFK_WOEBNQ5RNsAd0n1PYmTcgfHjBQ?e=KQjMMt  

After you download the dataset and metadata files, your directory should look like:

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
  
For pre-training, this is the default command:  
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 /path/to/pretraining/main_pretrain.py \
    --batch_size 8 --accum_iter 16 \
    --norm_pix_loss --epochs 100 \
    --blr 1.5e-4 --mask_ratio 0.75 \
    --input_size 224 --patch_size 16 \
    --model mae_vit_large_patch16 \
    --model_type temporal \
    --dataset_type temporal \
    --train_path <PATH_TO_DATASET_ROOT_FOLDER>/train_62classes.csv
    --output_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --num_workers 8  

# Fine-tuning  
You can download the fine-training dataset for flood segmentation here: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/EmWszM8rUpdNsNeyKAjxmR8BbzmuHyqbGphc_6XDwLjTjQ?e=Mi1U6D  
After you download the dataset and metadata files, your directory should look like:

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
  
  For fine-tuning, this is the default command:  

  python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 /path/to/finetuning/main_finetune.py \
    --batch_size 8 --accum_iter 16 \
    --norm_pix_loss --epochs 100 \
    --blr 1.5e-4 --mask_ratio 0.75 \
    --input_size 224 --patch_size 16 \
    --model mae_vit_large_patch16 \
    --model_type temporal \
    --dataset_type temporal \
    --train_path <PATH_TO_DATASET_ROOT_FOLDER>/train_62classes.csv
    --output_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --num_workers 8


# Pre-trained Models
EvaMAE: https://indiana-my.sharepoint.com/:f:/g/personal/adhiksa_iu_edu/EjmSoZOUjY5PpdZJ2xfhOmcBY3-Vy5se16OB1xBYh1wiZw?e=TWIhoG  
