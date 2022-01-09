# post_processing_tool_for_instance_segmentation_on_SXT
A post processing tool for organelle instance segmentation on soft X-ray tomograms


## What the tool does

This is an intensity-based post-processing tool to segment organelle instances, which can segment spherical(insulin vesicles) and columnar(mitochondria) organelles on the basis of voxel intensity of the raw tomograms, semantic segmentation mask, and prior knowledge of the organelle morphology. 



## What the code consists of

intensity-based post-processing tool

├── vesicle_seg -- to segment insulin vesicles instances especially the vesicles which are attached with each other

├── mito_seg -- to segment mitochondria instances using k-means clustering

├── build_synthetic_benchmark -- to build synthetic benchmark

├── mAP -- to meassure the accuracy of the segmentation result



## Pipeline

figure1





## Tutorial

tutorial link



## contact

angdi@

liping@
