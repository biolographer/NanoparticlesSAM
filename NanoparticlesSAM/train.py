from dataset import *
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import torch.optim as optim
from tqdem import tqdm


import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_FOLDER = '/content/drive/MyDrive/shield_data/training data'
EPOCHS = 100000

data = dataset.CircleMaskDataset(
    image_dir=IMAGE_FOLDER,
    metadata_fn=dataset.get_circle_metadata,
    crop_banner=True,
    convert_to_tensor=True,
)


def read_batch(data): # read random image and its annotaion from  the dataset (LabPics)
    #  select image

     ent  = data[np.random.randint(len(data))] # choose random entry
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

# set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


# Load model

sam2_checkpoint = "/content/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint,
                  device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

#The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using: predictor.model.image_encoder.train(True)
#Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).

optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler(device, ) # mixed precision

# Training loop

for itr in tqdm(range(EPOCHS)):
    with torch.cuda.amp.autocast(device,): # cast to mix precision
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

            if itr%1000==0: torch.save(predictor.model.state_dict(), "model.torch");print("save model")

            # Display results

            if itr==0: mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)",itr, "Accuracy(IOU)=",mean_iou)