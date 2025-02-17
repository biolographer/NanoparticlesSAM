# SAM for SEM measurements of nanoparticles

This repository contains the codes of the [paper](https://www.nature.com/articles/s41598-025-86327-x): "Pre-Trained Artificial Intelligence-Aided Analysis of Nanoparticles Using the Segment Anything Model" published in Scientific Reports (2025). 

The original method uses the segment anything model to create segmentation masks of nanoparticles of SEM measurements. The code was adapted to measure size distributions of mixtures of spherical nanoparticles with surface roughness. Following features were added:

- More robust roundness calculation
    - The roundness of rough particles was underestimated due to the coarse perimeter of the masks
    - sklearn image was used to smoothen the surface before perimeter determination
- parsing of JEOL SEM image metadata
- integration of per-image pixel-to-nanometer conversion for JEOL SEM measurements
    - can be entered manually if known
- filtering according to physical particle size (in nanometer)


## Original Publications

* Gabriel A. A. Monteiro - Colloid Chemistry, Department of Chemistry, University of Konstanz, Germany. [Orcid](https://orcid.org/0000-0002-5049-1704)
* Bruno A. A. Monteiro - Pattern Recognition and Earth Observation Laboratory, Department of Computer Science, UFMG, Belo Horizonte, Brazil. [Orcid](https://orcid.org/0000-0001-7288-5504)
* Jefersson A. dos Santos - Department of Computer Science, University of Sheffield, United Kingdom. [Orcid](https://orcid.org/0000-0002-8889-1586)
* Alexander Wittemann - Colloid Chemistry, Department of Chemistry, University of Konstanz, Germany. [Orcid](https://orcid.org/0000-0002-8822-779X)


