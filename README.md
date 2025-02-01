# Pre-trained artificial intelligence-aided analysis of nanoparticles using the segment anything model

Complex structures can be understood as compositions of smaller, more basic elements. The characterization of these structures requires an analysis of their constituents and their spatial configuration. Examples can be found in systems as diverse as galaxies, alloys, living tissues, cells, and even nanoparticles. In the latter field, the most challenging examples are those of subdivided particles and particle-based materials, due to the close proximity of their constituents. The characterization of such nanostructured materials is typically conducted through the utilization of micrographs. Despite the importance of micrograph analysis, the extraction of quantitative data is often constrained. The presented effort demonstrates the morphological characterization of subdivided particles utilizing a pre-trained artificial intelligence model. The results are validated using three types of nanoparticles: nanospheres, dumbbells, and trimers. The automated segmentation of whole particles, as well as their individual subdivisions, is investigated using the Segment Anything Model, which is based on a pre-trained neural network. The subdivisions of the particles are organized into sets, which presents a novel approach in this field. These sets collate data derived from a large ensemble of specific particle domains indicating to which particle each subdomain belongs. The arrangement of subdivisions into sets to characterize complex nanoparticles expands the information gathered from microscopy analysis. The presented method, which employs a pre-trained deep learning model, outperforms traditional techniques by circumventing systemic errors and human bias. It can effectively automate the analysis of particles, thereby providing more accurate and efficient results.


This repository contains the codes of the [paper](https://www.nature.com/articles/s41598-025-86327-x): "Pre-Trained Artificial Intelligence-Aided Analysis of Nanoparticles Using the Segment Anything Model" submitted to Scientific Reports (2024). 

## Authored by:

* Gabriel A. A. Monteiro - Colloid Chemistry, Department of Chemistry, University of Konstanz, Germany. [Orcid](https://orcid.org/0000-0002-5049-1704)
* Bruno A. A. Monteiro - Pattern Recognition and Earth Observation Laboratory, Department of Computer Science, UFMG, Belo Horizonte, Brazil. [Orcid](https://orcid.org/0000-0001-7288-5504)
* Jefersson A. dos Santos - Department of Computer Science, University of Sheffield, United Kingdom. [Orcid](https://orcid.org/0000-0002-8889-1586)
* Alexander Wittemann - Colloid Chemistry, Department of Chemistry, University of Konstanz, Germany. [Orcid](https://orcid.org/0000-0002-8822-779X)

## Keywords

    image segmentation
    electron microscopy
    nanoparticles
    artificial intelligence
    segment anything model

## Key Ideas

We are expanding the results of the Segment Anything Model (Meta, 2023) to identify colloid particles in SEM images and compute morphological particle properties from it. 

We present an analysis routine based on a pre-trained deep learning model for image segmentation that allows us to determine the dimensions and morphology of structures with different levels of complexity depicted in micrographs.

Using a pre-trained neural network, the Segment Anything Model, this method shows automated segmentation between particles' subdivisions. 

From this stage on, subdivisions are organized into sets representing the particles, a novelty in the field.

The arrangement of subdivisions into sets to characterize complex nanoparticles expands the information gathered from microscopy analysis.

The model colloids used to test the method are compared to previously published results, demonstrating that the novel method avoids systemic errors and human bias.

## Model 

The presented analyses dismiss the need to train a dedicated deep learning-based model. This was achieved using a previously trained general segmentation model and optimizing post-processing techniques.

The segmentation masks are obtained using the Segment Anything Model, which is an extensively trained model. Our approach consists of using this model to obtain the masks and then post-processing them. In this post-process, the segmented elements are assigned to a complex particle, which allows properties to be obtained for both the individual elements and the whole particle.

The SAM model is available in their official [Repository](https://github.com/facebookresearch/segment-anything#installation). Checkpoint used: sam_vit_h_4b8939.pth Can be downloaded from the official project: [ModelCheckpoints](https://github.com/facebookresearch/segment-anything#installation)




## Dataset

Scanning electron microscope [(SEM) Images Dataset](https://kondata.uni-konstanz.de/radar/en/dataset/EsfTYSZxEqPwiVkZ?token=JkMlsbdRVNoyALehTOiy#) provided by the Colloid Chemistry, Department of Chemistry, University of Konstanz. Also available in this [repository](Dataset).


<img src="Dataset\SEM\GM 124_00.png" width="440" height="310">

## Results

Example of result for a trimer particle showing the masks obtained from the SAM with our post-processing techniques to focus on the relevant masks. Also, our technique allows us to determine which element belongs to which particle, enhancing the provided model output.   

<img src="SAM-BasedMethod\results\trimers\result_example_124_00.png" width="440" height="310">


## Funding
    Deutsche Forschungsgemeinschaft
    CAPES
    CNPq
    FAPEMIG
    Serrapilheira Institute
