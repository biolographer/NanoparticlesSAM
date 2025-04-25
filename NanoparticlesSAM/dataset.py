import cv2
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

def get_circle_from_points(p1, p2, p3):
    """
    Calculate circle center and radius given 3 points.
    """
    A = np.array([
        [p1[0] - p2[0], p1[1] - p2[1]],
        [p1[0] - p3[0], p1[1] - p3[1]]
    ])
    B = np.array([
        [(p1[0]**2 - p2[0]**2 + p1[1]**2 - p2[1]**2) / 2],
        [(p1[0]**2 - p3[0]**2 + p1[1]**2 - p3[1]**2) / 2]
    ])
    center = np.linalg.solve(A, B).flatten()
    radius = np.linalg.norm(center - np.array(p1))
    return tuple(center.astype(int)), int(radius)

def get_circle_metadata(filename):
    # functions for jeol tiff images

    with open(filename, '+rb') as f:
        data = f.read()
    textdata = data.decode('utf-16-le', errors='ignore')
    metadata_index = textdata.find('<Property Key="measure:ShapePoints" IsArray="true">')
    
    # if metadata is not in image
    if metadata_index == -1:
        print('\nparsing circle masks from image failed...')
        return None

    metadata = []
    meta_text = textdata.split('<Property Key="measure:ShapePoints" IsArray="true">')[1:]
    meta_text = [re.sub(r'[\t\r\n]', '', i.strip().split('</Property>')[0]) for i in meta_text if i != '']
    for line in meta_text:
        doubles = list(map(float, re.findall(r'<double>(.*?)</double>', line)))
        metadata.append(doubles)

    metadata = np.array(metadata).reshape(len(meta_text), 3, 2)
    return metadata


class CircleMaskDataset(Dataset):
    def __init__(self, image_dir, metadata_fn, transform=None, convert_to_tensor=True, crop_banner=True):
        all_paths = list(Path(image_dir).glob("*.tif"))
        self.metadata_fn = metadata_fn
        self.transform = transform
        self.convert_to_tensor = convert_to_tensor
        self.crop_banner = crop_banner
        self.crop_height = 960
        self.crop_width = 1280

        # Only keep images with valid metadata
        self.image_paths = []
        for path in all_paths:
            if metadata_fn(path) is not None:
                self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def crop_from_origin(self, img, mask):
        ch, cw = self.crop_height, self.crop_width
        img = img[:ch, :cw, ...]
        mask = mask[:ch, :cw, ...]
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        metadata = self.metadata_fn(img_path)

        image = cv2.imread(str(img_path))  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        for col, triplet in enumerate(metadata):
            center, radius = get_circle_from_points(*triplet)
            cv2.circle(mask, center, radius, color=col + 1, thickness=-1)

        if self.crop_banner:
            image, mask = self.crop_from_origin(image, mask)

        if self.convert_to_tensor:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

