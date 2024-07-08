# Pre-Trained Artificial Intelligence-Aided Analysis of Nanoparticles Using the Segment Anything Model

(Abstract) Complex structures can be interpreted as assemblies of smaller, simpler elements. The characterization of these structures involves analyzing their constituents and their spatial configuration. Examples are found in systems as diverse as galaxies, alloys, living tissues, cells, down to nanoparticles. In the latter field, subdivided particles and particle-based materials are among the most prominent. Such nanostructured materials are characterized using micrographs. Despite the importance of micrograph analysis, the extraction of quantitative data is often limited. The effort presented here demonstrates the morphological characterization of subdivided particles with a pre-trained artificial intelligence model. This method shows automated segmentation between subdivisions of particles using the Segment Anything Model, which is based on a pre-trained neural network. From this stage on, the subdivisions are organized into sets, which is a novelty in the field. These sets gather data derived from a large ensemble of specific particle domains and contain information to which particle each subdomain belongs. The arrangement of subdivisions into sets to characterize complex nanoparticles expands the information gathered from microscopy analysis. The results gained based on selected model colloids are compared to previously published results, demonstrating that the novel method avoids systemic errors and human bias.  


This repository contains thec codes of the [paper](insert_papaer_url): "Pre-Trained Artificial Intelligence-Aided Analysis of Nanoparticles Using the Segment Anything Model" submitted to Scientific Reports (2024). 

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