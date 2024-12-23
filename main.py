import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary

import os
import argparse

from config import Config

from functions import train_and_val, test, dataloaders, UltrasoundDataset
from models import ViTForSegmentation, SAM, UNet, FCN

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add argparser here
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="specify the model to use (vit, fcn, unet)", default="ViT")
parser.add_argument("model_summary", type=bool, help="prints model architecture if set to True", default=False)
args = parser.parse_args()


if args.model == "vit":
    model = ViTForSegmentation()
elif args.model == "unet":
    model = UNet()
elif args.model == "fcn":
    model = FCN()
else:
    raise Exception("Model must be either vit, fcn or unet")

# if summary is True
if args.model_summary:
    summary(model, (1, 420, 580))

# Data Preprocessing
masks = pd.read_csv('/kaggle/input/ultrasound-nerve-segmentation/train_masks.csv')

train_masks = masks.dropna().copy()
test_masks = masks[masks['pixels'].isnull()].reset_index(drop=True)

# Labelled files and masks
train_masks['file'] = train_masks.apply(lambda row: f"{row['subject']}_{row['img']}.tif", axis=1)
train_files = train_masks['file'].tolist()
train_masks['mask'] = train_masks.apply(lambda row: f"{row['subject']}_{row['img']}_mask.tif", axis=1)
train_masks = train_masks['mask'].tolist()

train_path = "/kaggle/input/ultrasound-nerve-segmentation/train"

train_files.sort()
train_masks.sort()

train_files = [os.path.join(train_path, image) for image in train_files]
train_masks = [os.path.join(train_path, masks) for masks in train_masks]

dataset = UltrasoundDataset(train_files, train_masks)
train_dataloader, val_dataloader, test_dataloader = dataloaders(dataset)

# Model training and testing

model = nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler(device)

model = train_and_val(model, train_dataloader, val_dataloader, optimizer, criterion, scaler)
test(model, test_dataloader, criterion)
