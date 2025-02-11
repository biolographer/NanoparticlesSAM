import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.color import label2rgb

from particle_seg import *


def plot_compare_centroids(df_pred, label_small, label_middle, label_large, img):
  """
  Plots and compares predicted centroids with labeled centroids on an image.

  Args:
      df_pred (pandas.DataFrame): A DataFrame containing predicted centroid data.
      label_small (object): An object containing labeled centroids for small particles.
      label_middle (object): An object containing labeled centroids for medium particles.
      label_large (object): An object containing labeled centroids for large particles.
      img (np.ndarray): The image to plot on.

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Extract x and y coordinates from predicted centroids
  y_plot = [x for x, y in df_pred.centroid]
  x_plot = [y for x, y in df_pred.centroid]

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(16, 9))

  # Display the image
  ax.imshow(img)

  # Plot labeled centroids with different colors and transparency
  ax.scatter(label_small.x_adj, label_small.y_adj, c='b', alpha=0.8, s=12, label='Small')
  ax.scatter(label_middle.x_adj, label_middle.y_adj, c='g', alpha=0.8, s=12, label='Medium')
  ax.scatter(label_large.x_adj, label_large.y_adj, c='r', alpha=0.8, s=12, label='Large')

  # Plot predicted centroids in red
  ax.scatter(x_plot, y_plot, c='r', alpha=0.8, s=12, label='Predicted')

  # Add legend
  plt.legend()

  # Display the plot
  plt.show()


def plot_lobes(df_lobes, comb_mask=None, img=None, save=False, save_path=None, name=None):
  """
  Plots and visualizes lobes identified in a DataFrame on an image (optional).

  Args:
      df_lobes (pandas.DataFrame): A DataFrame containing lobe data.
      comb_mask (np.ndarray, optional): A combined mask for segmentation (optional).
      img (np.ndarray, optional): The image to plot on (optional).
      save (bool, optional): Flag to save the plot (default: False).
      save_path (str, optional): Path to save the plot (if save is True).
      name (str, optional): Name for the plot (if save is True).

  Returns:
      matplotlib.figure.Figure: The created plot figure.
  """

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(16, 12))

  # Check if combined mask exists and display image with segmentation colors
  if comb_mask is not None and comb_mask.any():
    image_label_felzen = label2rgb(comb_mask, image=img)
    plt.imshow(image_label_felzen)
  # Otherwise, display the image without segmentation
  else:
    plt.imshow(img)

  # Iterate through unique lobe IDs and plot corresponding centroids
  for idx in df_lobes['ID'].unique():
    filtered_df = df_lobes[df_lobes['ID'] == idx]
    xx = [y for x, y in filtered_df.centroid]
    yy = [x for x, y in filtered_df.centroid]
    plt.scatter(xx, yy, alpha=0.85)

    # Iterate through unique lobes and plot labels for each centroid
    for lobe in filtered_df['lobe'].unique():
      lobe_df = filtered_df[filtered_df['lobe'] == lobe]
      xxx = [y for x, y in lobe_df.centroid]
      yyy = [x for x, y in lobe_df.centroid]
      plt.text(xxx[0], yyy[0], str(lobe), color='blue', fontsize=12)

  # Return the created plot figure
  return fig



def plot_dumbbells(dataframe, img, comb_mask=None, save=False, name=None, save_path=None):
  """
  Plots dumbbells identified in a DataFrame on an image (optionally with segmentation mask).

  Args:
      dataframe (pandas.DataFrame): A DataFrame containing dumbbell data.
      img (np.ndarray): The image to plot on.
      comb_mask (np.ndarray, optional): A combined mask for segmentation (optional).
      save (bool, optional): Flag to save the plot (default: False).
      name (str, optional): Name for the plot (if save is True).
      save_path (str, optional): Path to save the plot (if save is True).

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(16, 12))

  # Check if combined mask exists and display image with segmentation colors
  if comb_mask is not None and comb_mask.any():
    image_label_felzen = label2rgb(comb_mask, image=img)
    plt.imshow(image_label_felzen)  # TODO: Adjust alpha to visualize image beneath segmentation colors
  else:
    plt.imshow(img)

  # Iterate through unique dumbbell IDs and plot corresponding centroids with labels
  for idx in dataframe['ID'].unique():
    yy = [x for x, y in dataframe[dataframe['ID'] == idx].centroid]
    xx = [y for x, y in dataframe[dataframe['ID'] == idx].centroid]
    plt.scatter(xx, yy, alpha=0.85)
    for x, y in zip(xx, yy):
      plt.text(x, y, str(int(idx)), color='red', fontsize=12)  # Add text label for each centroid

  # Optionally save the plot
  if save:
    plt.savefig(f'{save_path}/dumbbells_{name}.png')

  plt.show()


def plot_tripods_numbered(img, dataframe):
  """
  Plots tripods identified in a DataFrame on an image with text labels for each centroid.

  Args:
      img (np.ndarray): The image to plot on.
      dataframe (pandas.DataFrame): A DataFrame containing tripod data.

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(16, 12))

  # Display the image
  plt.imshow(img)

  # Iterate through unique tripod IDs and plot corresponding centroids with labels
  for idx in dataframe['ID'].unique():
    yy = [x for x, y in dataframe[dataframe['ID'] == idx].centroid]
    xx = [y for x, y in dataframe[dataframe['ID'] == idx].centroid]
    plt.scatter(xx, yy, alpha=0.85)
    for x, y in zip(xx, yy):
      plt.text(x, y, str(idx), color='red', fontsize=12)  # Add text label for each centroid

  plt.show()


def plot_rect(img, dataframe, segmentation_segments=None, mask_available=False, bbox_col='bbox'):
  """
  Plots rectangles around bounding boxes of objects identified in a DataFrame on an image (optionally with segmentation mask).

  Args:
      img (np.ndarray): The image to plot on.
      dataframe (pandas.DataFrame): A DataFrame containing bounding box data.
      segmentation_segments (np.ndarray, optional): A segmentation mask (optional).
      mask_available (bool, optional): Flag indicating presence of segmentation mask (default: False).
      bbox_col (str, optional): The column name containing bounding box information in the DataFrame (default: 'bbox').

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Extract x and y coordinates from centroids
  y = [x for x, y in dataframe.centroid]
  x = [y for x, y in dataframe.centroid]

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(25, 18))

  # Display the image with segmentation mask if available
  if mask_available:
    image_label_overlay = label2rgb(segmentation_segments, image=img, bg_label=0)
    ax.imshow(image_label_overlay)
  else:
    ax.imshow(img)

  # Plot red scatter points at centroids
  plt.scatter(x, y, s=12, c='r')

  # Iterate through rows in the DataFrame and plot individual rectangles
  for idx in range(dataframe.shape[0]):
    min_row, min_col, max_row, max_col = dataframe.loc[idx, bbox_col]
    rectangle = mpatches.Rectangle(xy=(min_col, min_row),
                                 width=max_col - min_col,
                                 height=max_row - min_row,
                                 fill=False,
                                 edgecolor='red',
                                 linewidth=0.5)
    ax.add_patch(rectangle)

  plt.show()



def plot(image1, image2):
  """
  Plots two images side-by-side for comparison.

  Args:
      image1 (np.ndarray): The first image to plot.
      image2 (np.ndarray): The second image to plot.

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Create a figure with two subplots
  fig, axes = plt.subplots(1, 2, figsize=(16, 9))

  # Display the first image with title
  axes[0].imshow(image1, cmap='gray')
  axes[0].set_title('Original Image')

  # Display the second image with title
  axes[1].imshow(image2, cmap='gray')
  axes[1].set_title('Result')

  # Adjust spacing between subplots for better visualization
  plt.tight_layout()

  # Display the plot
  plt.show()


def plot_seg_mask(img, segmentation_segments, save=False, name=None,save_path=None, alpha=0.3):
  """
  Plots an image with its corresponding segmentation mask overlayed with adjustable transparency.

  Args:
      img (np.ndarray): The image to plot.
      segmentation_segments (np.ndarray): The segmentation mask.
      save (bool, optional): Flag to save the plot (default: False).
      name (str, optional): Name for the plot (if save is True).
      alpha (float, optional): Transparency level for the segmentation mask overlay (default: 0.3).

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(16, 10))

  # Create the segmentation mask overlay with specified alpha
  image_label_overlay = label2rgb(segmentation_segments, image=img, alpha=0.9)

  # Display the original image
  ax.imshow(img)

  # Overlay the segmentation mask with transparency
  ax.imshow(image_label_overlay, alpha=alpha)

  # Optionally save the plot
  if save:
    plt.savefig(f'{save_path}/{name}_seg_mask.png')  # Assume `save_path` is defined elsewhere

  plt.show()


def plot_single_mask(imagem, mask):
  """
  Plots an image alongside its corresponding mask.

  Args:
      imagem (np.ndarray): The image to plot.
      mask (np.ndarray): The mask to plot.

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Create a figure
  plt.figure(figsize=(16, 10))

  # Display the image
  plt.imshow(imagem)

  # Overlay the mask with a specific colormap and transparency
  plt.imshow(mask, alpha=0.7, cmap='plasma')

  # Turn off axis visibility
  plt.axis('off')

  plt.show()


def plot_overlay_mask(imagem, mask):
  """
  Plots an image with its corresponding mask overlaid in separate colormaps and transparencies.

  Args:
      imagem (np.ndarray): The image to plot.
      mask (np.ndarray): The mask to plot.

  Returns:
      None (This function displays a plot and does not return any value.)
  """

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(18, 16))

  # Display the original image
  ax.imshow(imagem)

  # Create the segmentation mask overlay with alpha
  image_label_overlay = label2rgb(mask, image=imagem, alpha=0.9)

  # Overlay the segmentation mask with one transparency level
  ax.imshow(image_label_overlay, alpha=0.5)

  # Overlay the original mask with another transparency and colormap
  ax.imshow(mask, alpha=0.5, cmap='viridis')

  # Turn off axis visibility
  plt.axis('off')

  plt.show()
