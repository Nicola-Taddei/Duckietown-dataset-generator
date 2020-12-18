# Notebooks for Duckietown Segmentation
This folder contains Jupyter Notebooks that were used for different purpose on Duckietown.

## Dataset Conversion
The [Dataset Generator](../dataset_generator/README.md) generate data which is not directly compatible with existing segmentation model training loops. Most segmentation model support the MS COCO format, so we convert the "raw" dataset to MS COCO using the following notebooks:

- [duckie_to_coco.ipynb](duckie_to_coco.ipynb). (For basic segmentation)
- [duckie_to_coco_bezier.ipynb](duckie_to_coco_bezier.ipynb). (For basic segmentation + bezier lines)

The dataset can then be visualized (and trained!) in the Detectron2 framework. Detectron2 is not suitable for real-time, but can be used to check the upper bound for performance of a specific dataset, before training it with the "mobile segmentaion" model from Turkmen.

See: [detectron2_duckie.ipynb](detectron2_duckie.ipynb)

## Dataset Merging
Sometimes, it can be interesting to merge datasets together, for instance, a "real" dataset and a simulation dataset. The [merge_coco_dataset.ipynb](merge_coco_dataset.ipynb) notebooks does that. 

## Controller Exploration
The [controller_exploration.ipynb](controller_exploration.ipynb) notebooks allows to visualize and debug the pure pursuit controller using logs from the actual controller. 
