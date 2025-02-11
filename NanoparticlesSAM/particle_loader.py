import numpy as np
import cv2
import os


class Particle_Dataset:
  """
  This class represents a dataset of particle images for a specific particle type.

  Args:
      root (str): The root directory of the dataset.
      img_idx (int): The particle type code (e.g., 124 for trimers, 116 for dumbbells, 59 for spheres).
  """

  def __init__(self, root, device=None, crop_banner=False):
    """
    Initializes the Particle_Dataset object.

    Args:
        root (str): The root directory of the dataset.
        img_idx (int): The particle type code.
    """

    self.root = root
    self.device = device

    if device == 'jeol':
       self.crop_banner = True
       print(f'Setting "crop banner={self.crop_banner}" due to {self.device} device')
    else:
       print(f'banner cropping set to {self.device}')

    self.files = self.make_dataset()

  def __len__(self):
    """
    Returns the number of images in the dataset.

    Returns:
        int: The number of images in the dataset.
    """

    return len(self.files)

  def make_dataset(self):
    """
    Creates a list of filenames of the particle images based on the particle type code.

    Returns:
        list: A list of filenames of the particle images.
    """

    # Define data path
    files = os.listdir(self.root)  # Get all files in the path
    # print(files)
    # Filter files based on extension and particle type code
    filtered_files = sorted([r for r in files if 'tif' in r.split('.')[-1]])
    
    #filtered_imgs = sorted([r for r in filtered_files if r.split('_')[0].split(' ')[1] == str(self.img_idx)])
    return filtered_files

  def get_idx_crop(self, image, sample):
    """
    Crops the image and identifies the index for further processing based on the particle type.

    Args:
        image (np.ndarray): The raw image data.
        sample (str): The filename of the image.

    Returns:
        int or list: The index for further processing depending on the particle type.
    """

    cropped_img = image[2:-2, 2:-2, :]  # Crop the image

    # Find and analyze white pixels (background)
    i, j, k = np.where(cropped_img < 2)
    unique_white, counts_white = np.unique(i, return_counts=True)
    # print(sample.split('_')[1],sample.split('_'))
    # Determine the index based on particle type
    if sample.split('_')[0].split(' ')[1] != '59':
      return unique_white[0]  # Return the first white pixel index for non-spherical particles
    else:
      return list(counts_white).index(max(counts_white))  # Return the index of the most frequent white pixel for spheres

  def __getitem__(self, index):
    """
    Retrieves a specific image and its name from the dataset.

    Args:
        index (int): The index of the image to retrieve.

    Returns:
        tuple: A tuple containing the cropped image (np.ndarray) its filename (str), and it's metadata (dict).
    """
    particle_idx = self.files[index]
    particle_name = str(particle_idx).split("/")[-1]
    filename = os.path.join(self.root, particle_name)
    
    if self.device == None:
      image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
      cropped_img = image[2:-2, 2:-2, :]

      # Apply cropping based on the particle type and retrieved index
      cropped_img = cropped_img[:self.get_idx_crop(image, particle_name), :, :]
      metadata = None
    elif self.device == 'jeol':
       cropped_img, metadata = self._parse_jeol_tiff(filename)

    return cropped_img, particle_name, metadata
  
  # functions for jeol tiff images
  def _parse_jeol_metadata(self, filename):
      with open(filename, '+rb') as f:
          data = f.read()
      textdata = data.decode('utf-16-le', errors='ignore')

      metadata_index = textdata.find('$CM_FORMAT')
      raw_metadata = textdata[metadata_index:].split('$')

      metadata = {}

      for entry in raw_metadata:
          if entry:
              key_value = [i.strip() for i in entry.split('=')]
              metadata[key_value[0]] = key_value[1]
      
      return metadata

  def _parse_jeol_tiff(self, filename):
      img = cv2.imread(filename)
      metadata = self._parse_jeol_metadata(filename)

      if self.crop_banner:
          col_nr, row_nr = [int(i) for i in metadata['CM_IMAGE_SIZE'].split(' ')]
          img = img[:row_nr, :col_nr, :]

      return (img, metadata)
