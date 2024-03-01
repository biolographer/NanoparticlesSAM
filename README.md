# Particle Segmentation and Morphology Analysis using Segment Anything Model (SAM)

Expanding the results of the Segment Anything Model (Meta, 2023) to identify colloid particles in SEM images and compute morphological particle properties from it. 

We present an analysis routine based on a pre-trained deep learning model for image segmentation that allows determining the dimensions and morphology of structures with different levels of complexity depicted in micrographs.

Codes of the [paper](insert_papaer_url): "Automated Morphology Analysis of Nanoparticles Using the Segment Anything Pre-Trained Artificial Intelligence Model" submitted to Scientific Reports (2024). 

Authored by:
* Gabriel A. A. Monteiro - Colloid Chemistry, Department of Chemistry, University of Konstanz, Germany
* Bruno A. A. Monteiro - Pattern Recognition and Earth Observation Laboratory, Department of Computer Science, UFMG, Belo Horizonte, Brazil
* Jefersson A. dos Santos - Department of Computer Science, University of Sheffield, United Kingdom
* Alexander Wittemann - Colloid Chemistry, Department of Chemistry, University of Konstanz, Germany

## Key Ideia

This method shows automated segmentation between particles' subdivisions using a pre-trained neural network, the Segment Anything Model. 

From this stage on, subdivisions are organized into sets representing the particles, a novelty in the field.

The arrangement of subdivisions into sets to characterize complex nanoparticles expands the information gathered from microscopy analysis.

The model colloids used to test the method are compared to previously published results, demonstrating that the novel method avoids systemic errors and human bias.

## Model 

The presented analyses dismiss the need to train a dedicated deep learning-based model. This was achieved using a previously trained general segmentation model and optimizing post-processing techniques.

The segmentation masks are obtained using the Segment Anything Model, which is an extensively trained model. Our approach consists on using this model to obtain the masks and then post-processing it. In this post-process, the segmented elements are assigned to a complex particle, which allows to obtain properties for both the indivudual elements and the whole particle.

The SAM model is avaiable in their official [Repository](https://github.com/facebookresearch/segment-anything#installation). Checkpoint used: sam_vit_h_4b8939.pth Can be downloaded from the official project: [ModelCheckpoints](https://github.com/facebookresearch/segment-anything#installation)




## Dataset

Scanning electron microscope [(SEM) Images](https://cloud.uni-konstanz.de/index.php/s/ajGGXeKxm4PYkjg?path=%2FDataset) provided by the Colloid Chemistry, Department of Chemistry, University of Konstanz. Also avaible in this [repository](Dataset).


<img src="Dataset\SEM\GM 124_00.png" width="340" height="200">

## Results

Example of result for a trimer particle showing the masks obtained from the SAM with our post-processing techniques to focus on the relevant masks. Also, our technique allows to determine which element belongs with which particle, enhancing the provided model output.   

<img src="SAM-BasedMethod\results\trimers\result_example_124_03.png" width="340" height="200">
