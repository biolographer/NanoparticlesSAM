# Particle Segmentation and Morphology Analysis using the SAM: a walkthrough

Here, we provide an easy-to-follow implementation of our method. A Demo notebook for each particle type is also available in this repository or directly in [Colab](https://drive.google.com/file/d/1EA-tuy8JxlPsfaHJCIxxFr-0hhbGmvRD/view?usp=sharing).

## Remembering Key Ideas

We present an *analysis routine based on a pre-trained deep learning model for image segmentation* that allows us to determine the dimensions and morphology of structures with different levels of complexity depicted in micrographs.

Using a pre-trained neural network, the Segment Anything Model, this method shows automated segmentation between particles' subdivisions. From this stage on, subdivisions are organized into sets representing the particles, a novelty in the field.

The arrangement of subdivisions into sets to characterize complex nanoparticles expands the information gathered from microscopy analysis.

The model colloids used to test the method are compared to previously published results, demonstrating that the novel method avoids systemic errors and human bias.

## First step: Obtaining data

Since particle morphology plays a decisive role in the behavior of colloidal systems, an extensive morphological characterization must be conducted for any given sample. This analysis is commonly done using micrographs, either obtained from optical microscopy, scanning electron microscopy (SEM), transmission electron microscopy (TEM), or atomic force microscopy (AFM). 

This paper utilizes **Scanning Electron Microscope - [SEM Images](https://cloud.uni-konstanz.de/index.php/s/ajGGXeKxm4PYkjg?path=%2FDataset)** - provided by the Colloid Chemistry, Department of Chemistry, University of Konstanz. Also available in this [repository](Dataset). 

![Alt Text](/images_plots\SEM_example.png)
*Example of SEM Image* 

Regardless of how the images are obtained, there is a usual need to measure at least 500 particles to obtain statistically relevant results. This process is very time-consuming and prone to human bias and systemic errors if conducted manually, which promotes the development of reliable automated measuring techniques.

### The used images can be imported to your code via *!wget* the SEM.zip folder or accessed directly from the repository.

```python
img_idx='124' #* 124: Tirmer particle; 116: Dumbbell particle; 59: Sphere particle

data_path = os.path.join(os.getcwd(),'MorphologyAnalysisFromSegmentation', 'Dataset', 'SEM')

img_set = Particle_Dataset(root =data_path,img_idx=img_idx)
print("Analinsando",img_set.__len__(),'imagens')
```

## Second step: Obtaining the model

Regardless of the strategy used to obtain morphological information from the images, this information must allow for determining parameters like particle size and particle size distribution. The segmentation based on traditional methodologies may produce inaccurate segmentation labels and fail to identify regions with more than one element. Such problems may be overcome using modern approaches like **deep learning** methodologies.

The effectiveness of these models relies on the multi-level representations of raw data via non-linear transformations that allow encoding semantic representations of that data that can be used for pattern recognition. Nevertheless, the performance of deep learning models is highly correlated to the amount and quality of data available during the training phase. Labeling data can be a slow, time-consuming, and biased process. There is an increasing interest in generalizing tasks to where only some images are available for training. 

The segment anything model (SAM) is a powerful method trained over a vast and diverse basis of annotated segmentation tasks. Thus, SAM can achieve high-quality semantic segmentation, which results in several tasks without domain-specific pre-training. The SAM model is available in their official [Repository](https://github.com/facebookresearch/segment-anything#installation). Checkpoint used: sam_vit_h_4b8939.pth Can be downloaded from the official project: [ModelCheckpoints](https://github.com/facebookresearch/segment-anything#installation)


![Alt Text](/images_plots\demonstrations\SAMDemo.gif)
*Demonstration of SAM segmenting landscape. Source: https://segment-anything.com/*

### This model can be easily used in Colab (demonstrated in the demo) or locally (below).

```python
sam_checkpoint = "sam_vit_h_4b8939.pth"

device = "cuda"
model_type = "vit_h"

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

```

## Third step: Running the model and post-processing it

To provide more accurate results, it was necessary to define some model hyperparameters empirically. This is due to the fact that the SAM is a generalist model and might require some parameter search to obtain the best results for each type of image. 

```python
#Set model Hiperparameters
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=36,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.80,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=350,  # Requires open-cv to run post-processing
)
```

After defining the parameters, we apply the SAM to our images. However, as the SAM is a model that can segment everything in the image, it was necessary to filter some of the obtained masks. These filters are mostly related to the position, size, and circularity of the provided masks and can be deeply checked in the [repository](SAM-BasedMethod\particle_seg.py). An example of a filtered result is shown below.

```python
print(f'{idx}/{img_set.__len__()}')

#Get image
img, name = img_set.__getitem__(idx)
saving_name = name.split('.')[0]
print(f'Analizing particle: {saving_name}')

#Apply the SAM method and filters 
comined_mask, simple_mask, dataframe_SAM = SAM_analysis(img,mask_generator)

```

![Alt Text](images_plots\demonstrations\mask_124_00.png)
*Example of masks after filtering.*

## Fourth step: Assigning elements to particles and characterizing them

We analyze the particles and their subdivisions and extract morphological information that is only obtainable when these analyses are combined. However, a method needs to be implemented to perform the complete characterization necessary for the particle analysis. 

To achieve this, we approach this task as an optimization problem, where the assignment of the element to the particle is done by minimizing the sum of distances between the centroids of the elements. 

With the provided post-processed masks and the assignment of each lobe to a particle, we can use Skimage to obtain the properties for each mask, which can then be analyzed as typical tabular data.

```python
#Combine masks
comb_mask = get_cob_mask_from_sam(dataframe_SAM)

#Get properties from the predicted masks
sam_df = label_props_SAM(img,comb_mask)

#Define which lobe belongs to which particle
df_id_particle = define_id_particle_from_regionprops(sam_df)

#Define which lobe is which
triplet_all = assign_lobes(df_id_particle)
triplet_all['img_name'] = name
#Save results
triplet_all.to_excel(f'{save_path}/df_analized_{saving_name}.xlsx')
#Plot
plot_lobes(triplet_all,comb_mask,img,save=True,name=saving_name)
```
![Alt Text](/SAM-BasedMethod\results\trimers\result_example_124_00.png)
*Example of final masks after filtering and assigning the lobes and particles to which they belong.*


## Final Remarks 

By comparing the results obtained using the model presented here to those obtained traditionally, we could also validate the data and show that the acceleration of the analysis does not come at the expense of accuracy. 

Despite our method assigning only some elements to each particle (due to problems in the SAM masks or the optimization step), it provides fast, accurate, and voluminous morphological data.

The segmentation assigns subdivided parts to the elements to which they belong, allowing for extracting more information from a single image. The development presented here can expand and accelerate the capabilities of the methods present in the literature while making them more accurate. 

