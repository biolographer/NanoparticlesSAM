import math
import numpy as np
import pandas as pd
import cv2

from skimage.measure import label, regionprops, find_contours
from scipy.spatial.distance import cdist


def SAM_analysis(img, mask_generator, size_border=40):
  """
  Analyzes an image using a mask generator and performs filtering based on area, circularity, and border proximity.

  Args:
      img (np.ndarray): The image to analyze.
      mask_generator (object): An object that generates masks for the image.
      size_border (int, optional): Size of the border region to exclude (default: 40).

  Returns:
      tuple: A tuple containing three elements:
          - combined_mask (np.ndarray): The combined mask after filtering.
          - combined_array (np.ndarray): The combined segmentation array after filtering.
          - filtered_df (pandas.DataFrame): The filtered DataFrame containing particle data.
  """

  # Generate masks using the mask generator
  masks = mask_generator.generate(img)

  # Convert masks to a DataFrame
  df = pd.DataFrame(masks)

  # Calculate additional features for each particle
  df['perimeter'] = df['segmentation'].apply(compute_perimeter)
  df['circularity'] = circularity(df['area'], df['perimeter'])
  df['point_coords_tuple'] = df['point_coords'].apply(lambda x: tuple(x[0]))

  # Filter particles based on circularity
  filtered_df = df[df['circularity'] > 0.35]

  # Filter particles based on area using interquartile range (IQR)
  q5, q95 = filtered_df.area.quantile([0.05, 0.95])
  filtered_df = filtered_df[(filtered_df['area'] < q95) & (filtered_df['area'] > q5)]

  # Sort and reset index for convenience
  filtered_df.sort_values(by='circularity', ascending=False, inplace=True)
  filtered_df.reset_index(inplace=True, drop=True)

  if size_border > 0:
    # Exclude particles near the border
    filtered_df = remove_border_particles(filtered_df, size=size_border)
    filtered_df.reset_index(inplace=True, drop=True)

  # Combine segmentation arrays for remaining particles
  filtered_combined_array = filtered_df.loc[0, 'segmentation']
  for idx in range(1, len(filtered_df)):
    filtered_combined_array += filtered_df.loc[idx, 'segmentation']

  # Create a combined mask with unique labels for each remaining particle
  comb_mask = np.zeros(filtered_df.segmentation[0].shape)
  for idx in range(len(filtered_df.segmentation)):
    comb_mask += np.where(filtered_df.loc[idx, 'segmentation'] == True, idx + 1, 0)

  return comb_mask, filtered_combined_array, filtered_df #comined_mask, simple_mask, dataframe_SAM
  

def sphere_segmentation(img, mask_generator, 
                        nanometer_per_pixel=None, 
                        diameter_cutoff=None,
                        circularity_cutoff = 0.75,
                        border_cutoff=True):
  """
  Analyzes an image using a mask generator and performs filtering based on area, circularity, and border proximity.

  Args:
      img (np.ndarray): The image to analyze.
      mask_generator (object): An object that generates masks for the image.
      nanometer_per_pixel (float): the size of a pixel in nm.
      diameter_cutoff (float): the particle size cutoff. 
      border_cutoff (bool): will remove particles that are < than their diameter away from border.

  Returns:
      tuple: A tuple containing three elements:
          - combined_mask (np.ndarray): The combined mask after filtering.
          - combined_array (np.ndarray): The combined segmentation array after filtering.
          - filtered_df (pandas.DataFrame): The filtered DataFrame containing particle data.
  """
  if not nanometer_per_pixel:
     nanometer_per_pixel = 1
     print(f'no value for the dimensions of a pixel in nanometer are given.\n Assuming 1 pixel = {nanometer_per_pixel}nm')
  else:
     print(f'1 pixel = {nanometer_per_pixel} nm')

  if not diameter_cutoff:
     diameter_cutoff = 0
     print(f'expected particle diameter automatically set to {diameter_cutoff} nm')
  else: 
     print(f'expected particle diameter = {diameter_cutoff} nm')
     nanometer_radius_cutoff = diameter_cutoff / 2.0
     pixel_radius_cutoff = nanometer_radius_cutoff / nanometer_per_pixel
     pixel_area_cutoff = np.pi*pixel_radius_cutoff**2
     

  # Generate masks using the mask generator
  masks = mask_generator.generate(img)

  # Convert masks to a DataFrame
  df = pd.DataFrame(masks)

  # Calculate additional features for each particle
  df['smooth_mask'] = df['segmentation'].apply(smoothened_mask)

  df['perimeter'] = df['smooth_mask'].apply(compute_perimeter)
  df['smooth_area'] = df['smooth_mask'].apply(lambda x: x.sum())
  df["circularity"] = df.apply(lambda row: circularity2(row['smooth_mask']), axis=1)

  #df['circularity'] = circularity(df['area'], df['perimeter'])
  df['estimated_radius'] = df['area'].apply(lambda x: np.sqrt(x / np.pi))

  df['point_coords_tuple'] = df['point_coords'].apply(lambda x: tuple(x[0]))

  # Filter particles based on circularity
  filtered_df = df[df['circularity'] > circularity_cutoff]
  q5, q95 = filtered_df.area.quantile([0.05, 0.95])

  # Filter particles based on area using interquartile range (IQR)
  filtered_df = filtered_df[(filtered_df['area'] < q95) & (filtered_df['area'] > q5)]
  filtered_df = filtered_df[(filtered_df['area'] > pixel_area_cutoff)]


  # Sort and reset index for convenience
  filtered_df.sort_values(by='circularity', ascending=False, inplace=True)
  filtered_df.reset_index(inplace=True, drop=True)

  if border_cutoff:
    img_height, img_width, _ = img.shape
    # Exclude particles near the border
    filtered_df = remove_border_particles(filtered_df, img_height, img_width)
    filtered_df.reset_index(inplace=True, drop=True)

  # Combine segmentation arrays for remaining particles
  filtered_combined_array = filtered_df.loc[0, 'segmentation']
  for idx in range(1, len(filtered_df)):
    filtered_combined_array += filtered_df.loc[idx, 'segmentation']

  # Create a combined mask with unique labels for each remaining particle
  comb_mask = np.zeros(filtered_df.segmentation[0].shape)
  for idx in range(len(filtered_df.segmentation)):
    comb_mask += np.where(filtered_df.loc[idx, 'segmentation'] == True, idx + 1, 0)

  return comb_mask, filtered_combined_array, filtered_df #comined_mask, simple_mask, dataframe_SAM


def remove_border_particles(df, height, width):
    """
    Removes particles whose center is closer than their radius to the image border.

    Args:
        df_test (pd.DataFrame): DataFrame containing 'radius' and 'point_coords_tuple' columns.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        pd.DataFrame: Filtered DataFrame with border particles removed.
    """
    def is_near_border(row):
        x, y = row["point_coords_tuple"]
        r = row["estimated_radius"]
        return (x - r < 0) or (y - r < 0) or (x + r > width) or (y + r > height)

    # Apply the filter
    df_filtered = df[~df.apply(is_near_border, axis=1)].copy()

    print(f"Removed {len(df) - len(df_filtered)} particles near the border.")
    return df_filtered
    

def compute_perimeter(segmentation_mask):
  """
  Calculates the perimeter of a particle represented by a segmentation mask.

  Args:
      segmentation_mask (list): A list representing the segmentation mask of a particle.

  Returns:
      float: The perimeter of the particle.
  """

  # Convert the list to a NumPy array
  mask_array = np.array(segmentation_mask)

  # Find contours in the mask
  contours = find_contours(mask_array, 0.5, fully_connected='high')

  # Calculate and return the total perimeter by summing individual contour lengths
  perimeter = np.sum([c.shape[0] for c in contours])
  return perimeter

# def circularity(area,perimeter):
#      return (4 * np.pi * area) / (perimeter ** 2)

def circularity(area, perimeter):
  """
  Calculates the circularity of a particle using its area and perimeter.

  Args:
      area (float): The area of the particle.
      perimeter (float): The perimeter of the particle.

  Returns:
      float: The circularity of the particle (4 * pi * area / perimeter^2).
  """

  return (4 * np.pi * area) / (perimeter ** 2)


def label_props_SAM(img, segmentation_segments):
  """
  Extracts and filters features from a segmented image using region properties.

  Args:
      img (np.ndarray): The image used to calculate intensity-based features.
      segmentation_segments (np.ndarray): The segmentation mask.

  Returns:
      pandas.DataFrame: A DataFrame containing filtered features for each particle.
  """

  # Define properties to extract
  properties = ['area', 'convex_area', 'mean_intensity', 'label', 'perimeter',
                'centroid', 'axis_major_length', 'feret_diameter_max',
                'orientation', 'solidity', 'eccentricity', 'bbox']

  # Create an empty DataFrame to store features
  dados_sam = pd.DataFrame(columns=properties)

  # Apply region labeling and extract properties
  labeled_sam = label(segmentation_segments)
  regions_sam = regionprops(labeled_sam, intensity_image=img)

  # Add extracted features row-wise to the DataFrame
  for region in regions_sam:
    measurements = [getattr(region, prop) for prop in properties]
    dados_sam.loc[len(dados_sam)] = measurements

  # Calculate circularity
  dados_sam['circularity'] = circularity(dados_sam['area'], dados_sam['perimeter'])

  # Apply area filtering based on specified min and max thresholds
  filtered_df = dados_sam[(dados_sam['area'] < 10000) & (dados_sam['area'] > 300)]

  # Apply area filtering based on IQR (Interquartile Range)
  q5, q95 = filtered_df.area.quantile([0.05, 0.95])
  filtered_df = filtered_df[(filtered_df['area'] < q95) & (filtered_df['area'] > q5)]

  # Apply circularity filtering
  filtered_df = filtered_df[filtered_df['circularity'] > 0.35]

  # Sort and reset index for convenience
  filtered_df.sort_values(by='circularity', ascending=False, inplace=True)
  filtered_df.reset_index(inplace=True, drop=True)

  return filtered_df


def get_comb_mask_from_sam(dataframe_SAM):
  """
  Creates a combined mask from a DataFrame containing segmentation information.

  Args:
      dataframe_SAM (pandas.DataFrame): A DataFrame containing segmentation data,
                                       including a 'segmentation' column for masks
                                       and a 'predicted_iou' column.

  Returns:
      np.ndarray: A NumPy array representing the combined mask.
  """

  # Copy the input DataFrame
  df_over = dataframe_SAM.copy()

  # Sort particles based on 'predicted_iou' (assuming higher is better)
  df_over.sort_values(by='predicted_iou', inplace=True, ascending=False)

  # Reset the index to ensure sequential labeling
  df_over.reset_index(inplace=True, drop=True)

  # Create a zero-filled array for the combined mask
  comb_mask = np.zeros(df_over.segmentation[0].shape)

  # Iterate through segmentation masks and combine them
  for idx in range(len(df_over.segmentation)):
    comb_mask += np.where(df_over.loc[idx, 'segmentation'] == True, idx + 1, 0)  # Use +1 for label starting at 1

  return comb_mask


def define_id_particle_from_regionprops(sam_df):
  """
  Assigns unique IDs to particles in a DataFrame based on spatial proximity.

  Args:
      sam_df (pandas.DataFrame): A DataFrame containing particle data,
                                 including a 'centroid' column for coordinates.

  Returns:
      pandas.DataFrame: A DataFrame with an additional 'ID' column for unique particle IDs.
  """

  # Copy the input DataFrame
  df_dist = sam_df.copy()

  # Sort by 'centroid' for proximity-based grouping
  df_dist.sort_values(by='centroid', inplace=True)
  df_dist.reset_index(inplace=True, drop=True)

  # Extract coordinates and calculate pairwise distances
  coords = df_dist.loc[:,'centroid']
  distance_matrix =cdist(coords.tolist(), coords.tolist())
  # Find nearest neighbors (top 3) for each particle
  idx = distance_matrix.argsort(axis=1)


  df_dist['candidate'] =''
  for row in range(df_dist.shape[0]):
      df_dist.loc[row,'candidate'] = tuple( ([idx[row,0]],[idx[row,1]],[idx[row,2]]) )
  
  # Assign candidate IDs and calculate total distances
  df_dist['sum_dist'] =''
  for row in range(df_dist.shape[0]):
      i,j,k = df_dist.loc[row,'candidate']
      df_dist.loc[row,'sum_dist'] =np.sum( [distance_matrix[row,i],distance_matrix[row,j],distance_matrix[row,k] ])



  # Sort by total distance (potentially indicating stronger grouping)
  df_sorted = df_dist.sort_values('sum_dist')

  # Assign unique IDs based on most frequent candidate combinations (potential clusters)
  df_sorted['best_cand'] = ''
  for index, row in df_sorted.iterrows():
    candidate = tuple(row['candidate'])
    min_dist = row['sum_dist']
    best_cand = candidate

    for i_2 in range(df_sorted.shape[0]):
      cand = df_sorted.loc[i_2,'candidate']
      distance = df_sorted.loc[i_2,'sum_dist']
      #check intersection
      if set([index]).intersection(set(cand)):
          #if it does, get the one with smaller sum of distances
          if distance<min_dist:
              min_dist = distance
              best_cand =cand

      df_sorted.at[index,'best_cand'] = list(best_cand)

  # Extract unique candidate combinations and their corresponding rows
  cands = df_sorted.best_cand
  indices_cand = cands.apply(pd.Series).stack().reset_index(drop=True).unique()

  # Identify valid candidate combinations with a frequency of 3 (potential clusters)
  df_cand = df_sorted[df_sorted.index.isin(indices_cand)]
  df_cand.sort_values('best_cand')
  indices = df_cand['best_cand'].isin(df_cand['best_cand'].value_counts()[df_cand['best_cand'].value_counts()==3].index).index
  indices = indices.unique()
  df_cand['best_cand'].value_counts()==3

  # Get the lists where the value counts are equal to 3
  valid_lists = df_cand['best_cand'].value_counts()[df_cand['best_cand'].value_counts() == 3].index.tolist()

  # Filter the DataFrame based on the valid lists
  filtered_df = df_cand[df_cand['best_cand'].isin(valid_lists)]

  filtered_df['ID'] = pd.factorize(filtered_df['best_cand'].apply(tuple))[0]
  df_final = filtered_df.sort_values('ID')
  return df_final


def middle_angle_between_points(point_a, point_b, point_c):
  """
  Calculates the middle angle between three points using vector dot product and arctangent.

  Args:
      point_a (object): An object with a 'centroid' attribute containing (x, y) coordinates.
      point_b (object): An object with a 'centroid' attribute containing (x, y) coordinates.
      point_c (object): An object with a 'centroid' attribute containing (x, y) coordinates.

  Returns:
      float: The middle angle between the three points in degrees.
  """

  # Calculate vectors from point B to points C and A
  vector_bc = (point_b.centroid[0] - point_c.centroid[0], point_b.centroid[1] - point_c.centroid[1])
  vector_ba = (point_b.centroid[0] - point_a.centroid[0], point_b.centroid[1] - point_a.centroid[1])

  # Calculate dot product and magnitude product of vectors
  dot_product = np.dot(vector_bc, vector_ba)
  magnitude_product = np.linalg.norm(vector_bc) * np.linalg.norm(vector_ba)

  # Calculate the angle in radians and convert to degrees
  angle_rad = np.arccos(dot_product / magnitude_product)
  angle_deg = math.degrees(angle_rad)

  return angle_deg


def get_max_angle(point1, point2, point3):
  """
  Finds the largest angle and its corresponding index among three middle angles.

  Args:
      point1 (object): An object with a 'centroid' attribute containing (x, y) coordinates.
      point2 (object): An object with a 'centroid' attribute containing (x, y) coordinates.
      point3 (object): An object with a 'centroid' attribute containing (x, y) coordinates.

  Returns:
      tuple: A tuple containing the maximum angle (degrees) and its index (0, 1, or 2).
  """

  # Calculate middle angles for each combination of points
  angle_123 = middle_angle_between_points(point1, point2, point3)
  angle_231 = middle_angle_between_points(point2, point3, point1)
  angle_312 = middle_angle_between_points(point3, point1, point2)

  # Ensure triangle angles sum to approximately 180 degrees for verification
  np.testing.assert_almost_equal(np.sum([angle_123, angle_231, angle_312]), 180, decimal=5)

  # Find maximum angle and its index
  max_angle = max(angle_123, angle_231, angle_312)
  max_angle_index = np.argmax([angle_123, angle_231, angle_312])

  return max_angle, max_angle_index


def set_lobe(df_particle, point1, point2, point3, middle_point, max_angle):
    """
    Assigns lobe labels and maximum angle to a DataFrame based on three points and a middle point.

    **Note:** The specific logic for assigning labels based on area comparisons and middle point index is currently undefined and requires further implementation.

    Args:
    df_particle (pandas.DataFrame): A DataFrame containing particle data.
    point1 (object): An object with a 'centroid' attribute containing (x, y) coordinates.
    point2 (object): An object with a 'centroid' attribute containing (x, y) coordinates.
    point3 (object): An object with a 'centroid' attribute containing (x, y) coordinates.
    middle_point (int): Index of the middle point (0, 1, or 2).
    max_angle (float): The maximum angle (degrees).

    Returns:
    pandas.DataFrame: The updated DataFrame with assigned lobe labels and maximum angle (placeholders for now).
    """
      
    indexes = df_particle.index.tolist()
    
    
    if middle_point == 0:
        lobe_b = point1
        df_particle.at[indexes[0], 'lobe'] = 'b'
        df_particle.at[indexes[0], 'max_angle'] = max_angle
        if df_particle.at[indexes[1], 'area'] > df_particle.at[indexes[2], 'area']:
            lobe_a = point2
            df_particle.at[indexes[1], 'lobe'] = 'a'
            lobe_c = point3
            df_particle.at[indexes[2], 'lobe'] = 'c'
        else:
            lobe_a = point3
            lobe_c = point2
            df_particle.at[indexes[2], 'lobe'] = 'a'
            df_particle.at[indexes[1], 'lobe'] = 'c'
            
    elif middle_point == 1:
        lobe_b = point2
        df_particle.at[indexes[1], 'lobe'] = 'b'
        df_particle.at[indexes[1], 'max_angle'] = max_angle
        if df_particle.at[indexes[0], 'area'] > df_particle.at[indexes[2], 'area']:
            lobe_a = point1
            lobe_c = point3
            df_particle.at[indexes[0], 'lobe'] = 'a'
            df_particle.at[indexes[2], 'lobe'] = 'c'
        else:
            lobe_a = point3
            lobe_c = point1
            df_particle.at[indexes[2], 'lobe'] = 'a'
            df_particle.at[indexes[0], 'lobe'] = 'c'
            
    elif middle_point == 2:
        lobe_b = point3
        df_particle.at[indexes[2], 'lobe'] = 'b'
        df_particle.at[indexes[2], 'max_angle'] = max_angle
        if df_particle.at[indexes[0], 'area'] > df_particle.at[indexes[1], 'area']:
            lobe_a = point1
            lobe_c = point2
            df_particle.at[indexes[0], 'lobe'] = 'a'
            df_particle.at[indexes[1], 'lobe'] = 'c'
        else:
            lobe_a = point2
            lobe_c = point1
            df_particle.at[indexes[1], 'lobe'] = 'a'
            df_particle.at[indexes[0], 'lobe'] = 'c'
    
    return df_particle


def assign_lobes(dataframe):
  """
  Assigns lobe labels and maximum angles to a DataFrame based on unique particle IDs.

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing particle data,
                                    including 'ID' column for unique identification.

  Returns:
      pandas.DataFrame: The updated DataFrame with 'max_angle' and 'lobe' columns
                        assigned for each particle.
  """

  # Add a column for 'max_angle' to store maximum angles
  dataframe['max_angle'] = ''

  # Create an empty DataFrame to store processed particles
  triplet_all = pd.DataFrame(columns=dataframe.columns)

  # Iterate through unique particle IDs
  for particle_id in dataframe['ID'].unique():
    # Filter particles belonging to the current ID
    particles = dataframe.loc[dataframe['ID'] == particle_id]

    # Extract coordinates for the three points associated with the particle ID
    point1 = particles.iloc[0, :]
    point2 = particles.iloc[1, :]
    point3 = particles.iloc[2, :]

    # Find the maximum angle and its corresponding middle point index
    max_angle, middle_point = get_max_angle(point1, point2, point3)

    # Assign lobe labels and maximum angles using the set_lobe function
    processed_particles = set_lobe(particles, point1, point2, point3, middle_point, max_angle)

    # Append the processed particles to the final DataFrame
    triplet_all = pd.concat([triplet_all, processed_particles])

  return triplet_all



def define_particle_lobes_from_regionprops(sam_df):
  """
  Defines particle lobes from a DataFrame containing region properties.

  Args:
      sam_df (pandas.DataFrame): A DataFrame containing region properties,
                                    likely obtained from skimage's regionprops.

  Returns:
      pandas.DataFrame: A DataFrame with assigned lobe information for each particle.
  """

  # Identify potential particle pairs based on defined criteria
  df_dist_init = define_candidates(sam_df)

  # Refine the potential pairs into final dumbbell-shaped particle pairs
  df_new = define_dumbbell_pairs(df_dist_init)

  # Calculate paired distances for the identified dumbbell-shaped pairs
  df_new = get_paired_distance(df_new)

  # Derive radius for each particle based on its major axis length
  df_new['radius'] = df_new['axis_major_length'] / 2

  # Assign lobe labels (e.g., 'lobe1', 'lobe2') to each particle in the pairs
  df_paired = assign_lobes_dumbbells(df_new)

  return df_paired



def define_candidates(dataframe):
  """
  Identifies potential particle pairs from a DataFrame containing centroid coordinates.

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing particle data,
                                    including a 'centroid' column with x-y coordinates.

  Returns:
      pandas.DataFrame: The updated DataFrame with a 'candidate' column
                        containing potential dumbbell particle pairs as tuples
                        (index of particle 1, index of particle 2, index of particle 3).
  """

  # Copy the DataFrame to avoid modifying the original
  df_dist_init = dataframe.copy()

  # Sort particles by their centroids (assuming 'centroid' has x, y coordinates)
  df_dist_init.sort_values(by='centroid', inplace=True)

  # Reset the index to ensure consistent indexing after sorting
  df_dist_init.reset_index(inplace=True, drop=True)

  # Extract the centroid coordinates into a NumPy array
  coords = df_dist_init.loc[:,'centroid']

  # Calculate the pairwise distances between all particles
  distance_matrix =cdist(coords.tolist(), coords.tolist())

  # Find the indices of the three closest neighbors (excluding itself) for each particle
  idx = distance_matrix.argsort(axis=1)

  # Create a 'candidate' column to store potential dumbbell pairs as tuples
  df_dist_init['candidate'] = ''


  # Assign potential dumbbell pairs (index of particle 1, index of particle 2, index of particle 3)
  for row in range(df_dist_init.shape[0]):
      df_dist_init.loc[row,'candidate'] = tuple( ([idx[row,1]],[idx[row,2]],[idx[row,3]]) )
  
  return df_dist_init



def set_n_closest_neighbor(dataframe, lista_indices, referencia, n):
  """
  Assigns particle pairs based on the n-th closest neighbor for each particle.

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing particle data,
                                    including a 'candidate' column with potential
                                    dumbbell particle pairs as tuples.
      lista_indices (list): A list of particle indices that still need to be paired.
      referencia (int): A reference number for assigning unique IDs to particle pairs.
      n (int): The index of the neighbor to consider for pairing (e.g., 1 for 2nd closest).

  Returns:
      tuple: A tuple containing the updated DataFrame, the remaining unpaired
              indices, and the updated reference number.
  """

  # Make a copy of the DataFrame and the list of indices to avoid modifying originals
  df = dataframe.copy()
  list_idx = lista_indices.copy()
  ref2 = referencia

  # Iterate over the remaining unpaired particles
  for row in list_idx:
    if row in list_idx:

      # Get the n-th closest neighbor of the current particle
      viz = df.loc[row, 'candidate']
      the_nth_closest = viz[n]

      # Check if the current particle is the first element in the neighbor's pair
      if row == df.loc[the_nth_closest, 'candidate'][0]:

        # Check if the distance between the particles is less than the sum of their radii
        if row  == df.loc[the_nth_closest,'candidate'][0]:
          # Assign the pair and update the particle IDs with a unique reference number
          df.at[row, 'par'] = [row, the_nth_closest]
          df.at[the_nth_closest, 'par'] = [row, the_nth_closest]
          df.at[row, 'ID'] = int(ref2)
          df.at[the_nth_closest, 'ID'] = int(ref2)

          # Remove the paired particles from the list of unpaired indices
          list_idx.remove(row)
          list_idx.remove(the_nth_closest)

          # Increment the reference number for the next pair
          ref2 += 1

  return df, list_idx, ref2


def define_dumbbell_pairs(dataframe):
  """
  Identifies dumbbell-shaped particle pairs from a DataFrame with potential pairs.

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing particle data,
                                    including 'candidate' and 'par' columns.
                                    'candidate' stores potential dumbbell pairs
                                    as tuples, and 'par' stores existing pairs.

  Returns:
      pandas.DataFrame: The updated DataFrame with refined particle pairs
                        identified as dumbbells.
  """

  # Copy the DataFrame to avoid modifying the original
  df_init = dataframe.copy()

  # Convert the index to a list for easier manipulation
  lista_idx = list(df_init.index)


  # Reference number for assigning unique IDs to pairs (starts at 0)
  ref = 0

  # Assign pairs based on the 1st closest neighbor using set_n_closest_neighbor
  df_init['par'] = ''
  df1, lista_idx, ref = set_n_closest_neighbor(df_init, lista_idx, ref, n=0)

  # Assign pairs based on the 2nd closest neighbor (excluding already paired particles)
  # using the updated DataFrame and remaining indices
  df2, lista_idx, ref = set_n_closest_neighbor(df1, lista_idx, ref, n=1)

  # Assign pairs based on the 3rd closest neighbor, following the same strategy
  df_new, lista_idx, ref = set_n_closest_neighbor(df2, lista_idx, ref, n=2)

  return df_new


def get_paired_distance(dataframe):
  """
  Calculates pairwise distances for particles identified as dumbbell pairs.

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing particle data,
                                    including 'par' and 'centroid' columns.
                                    'par' stores particle pairs as tuples,
                                    and 'centroid' stores x-y coordinates.

  Returns:
      pandas.DataFrame: The updated DataFrame with a 'distance' column
                        containing the pairwise distances for each pair.
  """

  # Copy the DataFrame to avoid modifying the original
  df_new = dataframe.copy()

  # Create a 'distance' column to store pairwise distances (initialized as empty strings)
  df_new['distance'] = ''

  # Iterate through each row of the DataFrame
  for row in df_new.index:
    # Check if the current particle has a paired partner
    if df_new.loc[row, 'par'] != '':

      # Extract indices of the paired particles
      d1, d2 = df_new.loc[row, 'par']

      # Get the centroid coordinates of the paired particles
      coords = df_new.loc[[d1, d2], 'centroid']

      # Calculate the pairwise distance matrix (2x2)
      distance_matrix = cdist(coords.tolist(), coords.tolist())

      # Assert that the calculated distance is the same regardless of index order
      assert distance_matrix[0][1] == distance_matrix[1][0], 'Distance mismatch'

      # Assign the calculated distance to the 'distance' column
      df_new.at[row, 'distance'] = distance_matrix[0][1]

  return df_new

def assign_lobes_dumbbells(dataframe):
  """
  Assigns lobe labels ("maior" and "menor") to particles in identified dumbbell pairs.

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing particle data,
                                    including 'ID', 'par', 'distance', 'radius',
                                    and 'area' columns.

  Returns:
      pandas.DataFrame: The updated DataFrame with assigned lobe labels
                        for each particle in a dumbbell pair.
  """

  # Filter the DataFrame to only include particles with identified pairs
  df_paired = dataframe[dataframe['par'] != '']

  # Create a 'lobe' column to store labels (initialized as empty strings)
  df_paired['lobe'] = ''

  # Iterate through unique particle IDs present in the paired DataFrame
  for particle_id in df_paired['ID'].unique():
    # Filter the paired DataFrame to get particles with the current ID
    df_particle = df_paired[df_paired['ID'] == particle_id]

    # Extract indices and distances of the two particles in the pair
    particle1_index, particle2_index = df_particle.index.tolist()
    particle1_distance, particle2_distance = df_particle['distance'].tolist()
    particle1_radius, particle2_radius = df_particle['radius'].tolist()

    # Check if the distances are less than the sum of their respective radii
    condition_met = (particle1_distance < particle1_radius + particle2_radius) and \
                    (particle2_distance < particle1_radius + particle2_radius)

    if condition_met:
      # Assign lobe labels based on area comparison
      if df_particle.loc[particle1_index, 'area'] > df_particle.loc[particle2_index, 'area']:
        df_paired.at[particle1_index, 'lobe'] = 'maior'  # Larger area
        df_paired.at[particle2_index, 'lobe'] = 'menor'  # Smaller area
      else:
        df_paired.at[particle1_index, 'lobe'] = 'menor'
        df_paired.at[particle2_index, 'lobe'] = 'maior'

  # Filter the paired DataFrame to keep only particles with assigned lobes
  df_paired = df_paired[df_paired['lobe'] != '']

  return df_paired


def smoothened_mask(mask):
    """Smooths the binary mask and returns its circularity."""

    # Ensure binary mask
    mask = (mask > 0).astype(np.uint8)

    # Morphological closing to fill small holes and smooth boundaries
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Gaussian blur to smooth rough edges
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

    # Re-threshold to keep it binary
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

    return mask

def circularity2(mask):
    # Compute circularity after smoothing
    region = regionprops(mask.astype(int))[0]  # Only one region per mask
    perimeter = region.perimeter
    area = region.area

    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return circularity
