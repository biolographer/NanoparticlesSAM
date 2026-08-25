import os
import sys
module_path = os.path.abspath(os.path.join('NanoparticlesSAM/NanoparticlesSAM'))
sys.path.append(module_path)

import argparse
from dataset import CircleMaskDataset, get_circle_metadata
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_args():
    parser = argparse.ArgumentParser(description="Paths and settings for SAM2 fine-tuning.")
    parser.add_argument('--TRAIN_IMAGE_FOLDER', type=str, required=True,
                        help='Path to the training image folder')
    parser.add_argument('--TEST_IMAGE_FOLDER', type=str, required=True,
                        help='Path to the testing image folder')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                        help='Path to the base SAM2 model checkpoint file')
    parser.add_argument('--CHECKPOINT_DIR', type=str, required=True,
                        help='Directory to save training checkpoints to')
    parser.add_argument('--LOG_FILE', type=str, required=True,
                        help='Path to the training log CSV file')
    return parser.parse_args()

args = get_args()

TRAIN_IMAGE_FOLDER = args.TRAIN_IMAGE_FOLDER
TEST_IMAGE_FOLDER = args.TEST_IMAGE_FOLDER
sam2_checkpoint = args.sam2_checkpoint
CHECKPOINT_DIR = args.CHECKPOINT_DIR
LOG_FILE = args.LOG_FILE

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"


EPOCHS = 2000

data = CircleMaskDataset(
    image_dir=TRAIN_IMAGE_FOLDER,
    metadata_fn=get_circle_metadata,
    crop_banner=True,
    convert_to_tensor=False,
)

test_data = CircleMaskDataset(
    image_dir=TEST_IMAGE_FOLDER,
    metadata_fn=get_circle_metadata,
    crop_banner=True,
    convert_to_tensor=False,
)

print(f'loaded training data, {len(data)} images')



def read_batch(data, index=None): # read random image and its annotaion from  the dataset (LabPics)
    #  select image
    if index == None:
        ent  = data[np.random.randint(len(data))] # choose random entry
    else:
        ent = data[index]
    Img = ent[0]  # read image
    particle_annotation = ent[1] # read annotation


    inds = np.unique(particle_annotation)[1:] # get the value for all masks
    points= []
    masks = []
    for ind in inds:
          mask = (particle_annotation == ind).astype(np.uint8) # make binary mask corresponding to index ind
          masks.append(mask)
          coords = np.argwhere(mask > 0) # get all coordinates in mask


          # filter out points too close to the edge of the circle
          miny, maxy, minx, maxx = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()

          cx = (minx + maxx) / 2
          cy = (miny + maxy) / 2

          radius = np.sqrt((maxx - minx)**2 + (maxy - miny)**2) / 2

          effective_radius = radius * 0.7

          distances = np.sqrt((coords[:, 1] - cx)**2 + (coords[:, 0] - cy)**2)

          point_mask = distances <= effective_radius
          filtered_coords = coords[point_mask]

          # choose random point/coordinate
          yx = np.array(filtered_coords[np.random.randint(len(filtered_coords))]) 
          points.append([[yx[1], yx[0]]])
          
    return Img,np.array(masks),np.array(points), np.ones([len(masks),1])

def validation_loop(val_data):
     val_loss = 0
     val_iou = 0
     with torch.no_grad(): 
        for i in range(len(val_data)):
            image, mask, input_point, input_label = read_batch(val_data, index=i)
            if mask.shape[0] == 0:
                continue  # Ignore empty batches
            predictor.set_image(image)  # Apply SAM image encoder to the image
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

            # Mask decoder
            batched_mode = unnorm_coords.shape[0] > 1  # Multi-object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image resolution

            # Segmentation Loss calculation (cross-entropy loss)
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            # Score loss calculation (Intersection over Union - IOU)
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

            val_loss += seg_loss + score_loss * 0.05  # Add to validation loss
            val_iou += np.mean(iou.cpu().detach().numpy())  # Add to IOU

        # Average validation results
        val_loss /= len(val_data)
        val_iou /= len(val_data)
        print(f"Validation - Step {itr}, Loss = {val_loss}, IOU = {val_iou}")
        return (val_loss, val_iou)     


# set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


# Load model


sam2_model = build_sam2(model_cfg, sam2_checkpoint,
                  device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

#The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using: predictor.model.image_encoder.train(True)
#Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).

optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.amp.GradScaler('cuda')

# Training loop

outfile = open(LOG_FILE, 'w')
outfile.write('step,loss,iou,stage\n')

for itr in tqdm(range(EPOCHS)):
    with torch.amp.autocast(device_type='cuda'): # cast to mix precision
            image,mask,input_point, input_label = read_batch(data) # load data batch
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image(image) # apply SAM image encoder to the image

            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder

            batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            if itr%200==0:
              torch.save(predictor.model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_{itr}.torch"))
              print("\n************\nsave model\n************\n")

            # Display results

            if itr==0: mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step",itr, "Accuracy(IOU)=",mean_iou)
            outfile.write(f'{itr},{loss},{mean_iou},train\n')


            if itr % 20==0 and itr > 1:
                print('____________________________________\n\tvalidating\n____________________________________\n')
                val_loss, val_iou = validation_loop(test_data)
                outfile.write(f'{itr},{val_loss},{val_iou},test\n')
                print('____________________________________\n')

outfile.close()