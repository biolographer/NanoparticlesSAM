from dataset import *
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_FOLDER = 'folder'

dataset = dataset.CircleMaskDataset(
    image_dir=IMAGE_FOLDER,
    metadata_fn=dataset.get_circle_metadata,
    crop_banner=True,
    convert_to_tensor=True,
)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)




sam = sam_model_registry["vit_b"](checkpoint="sam2_vit_b.pth")  # or vit_h/l
sam.train().cuda()




optimizer = optim.Adam(sam.parameters(), lr=1e-5)

def loss_fn(pred_mask, true_mask):
    return F.binary_cross_entropy_with_logits(pred_mask, true_mask)



for epoch in range(num_epochs):
    for imgs, masks in train_loader:
        imgs = imgs.cuda()         # shape: (B, 3, H, W)
        masks = masks.cuda()       # shape: (B, 1, H, W)

        # Forward pass through SAM (you may need to adapt based on how SAM2 handles images)
        outputs = sam(imgs)        # returns logits or masks

        pred_masks = outputs['masks']  # or similar key depending on the repo
        loss = loss_fn(pred_masks, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")