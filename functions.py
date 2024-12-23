import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from config import Config

config = Config


# Metric
def iou(pred, target):

    pred = pred.to(torch.int64)
    target = target.to(torch.int64)
    
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    iou = intersection / (union + 1e-6)
    
    return iou


# Custom Dataset
class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = config.transform

    def __len__(self):
        return len(self.image_paths)

    def _load_tif(self, path):
        with Image.open(path) as img:
            return np.array(img)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
    
        image = self._load_tif(img_path)
        mask = self._load_tif(mask_path)
    
        # Convert mask to binary
        mask = (mask > 0).astype(np.float32)
    
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
    
        image = torch.from_numpy(image).float() if not isinstance(image, torch.Tensor) else image.clone().detach().float()
        mask = torch.from_numpy(mask).float().unsqueeze(0) if not isinstance(mask, torch.Tensor) else mask.unsqueeze(0).clone().detach().float()
    
        return image, mask


# Dataloaders
def dataloaders(dataset):

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader, test_dataloader


# Training and Validation

def train_and_val(model, train_dataloader, val_dataloader, optimizer, criterion, scaler):

    val_iou_list = []

    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_iou = 0
        num_batches = 0

        # --- Train phase ----
        model.train()

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", unit="batch") as batch_progress:
            for batch in batch_progress:
                inputs, targets = batch
                inputs = inputs.cuda()
                targets = targets.cuda()

                if inputs.ndimension() == 3:
                    inputs = inputs.unsqueeze(1)  # Add channel dimension
                    targets = targets.unsqueeze(1)
                
                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()

                iou_metric = iou(outputs, targets)

                epoch_loss += loss.item()
                epoch_iou += iou_metric.item()
                num_batches += 1

                batch_progress.set_postfix(loss=epoch_loss/num_batches, iou=epoch_iou/num_batches)

        # --- Evaluation Phase ---
        model.eval()
        val_loss = 0
        val_iou = 0
        num_val_batches = 0

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_inputs, val_targets = val_batch
                val_inputs = val_inputs.cuda()
                val_targets = val_targets.cuda()

                if val_inputs.ndimension() == 3:
                    val_inputs = val_inputs.unsqueeze(1)
                    val_targets = val_targets.unsqueeze(1)

                with torch.amp.autocast("cuda"):
                    val_outputs = model(val_inputs)
                    val_loss_batch = criterion(val_outputs, val_targets)

                val_outputs = torch.sigmoid(val_outputs)
                val_outputs = (val_outputs > 0.5).float()

                val_iou_batch = iou(val_outputs, val_targets)

                val_loss += val_loss_batch.item()
                val_iou += val_iou_batch.item()
                num_val_batches += 1

        # Print epoch results
        val_iou_list.append(val_iou/num_val_batches)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {epoch_loss/num_batches:.4f}, 
            Train IoU: {epoch_iou/num_batches:.4f}, Val Loss: {val_loss/num_val_batches:.4f}, 
            Val IoU: {val_iou/num_val_batches:.4f}")
        
    return model
        
def test(model, test_dataloader, criterion):
    model.eval()
    test_loss = 0
    test_iou = 0
    num_test_batches = 0

    with torch.no_grad():
        for test_batch in test_dataloader:
            test_inputs, test_targets = test_batch
            test_inputs = test_inputs.cuda()
            test_targets = test_targets.cuda()

            if test_inputs.ndimension() == 3:
                test_inputs = test_inputs.unsqueeze(1)
                test_targets = test_targets.unsqueeze(1)

            with torch.amp.autocast("cuda"):
                test_outputs = model(test_inputs)
                test_loss_batch = criterion(test_outputs, test_targets)

            test_outputs = torch.sigmoid(test_outputs)
            test_outputs = (test_outputs > 0.5).float()

            test_iou_batch = iou(test_outputs, test_targets)

            test_loss += test_loss_batch.item()
            test_iou += test_iou_batch.item()
            num_test_batches += 1

    # Print epoch results
    print(f"Test Loss: {test_loss/num_test_batches:.4f}, Test IoU: {test_iou/num_test_batches:.4f}")