# How to prepare datasets.
The instructions in this file assume that you already downloaded or generated the raw datasets, and converted the dataset to a MS COCO format using one of the [Jupyter Notebooks in the notebooks folder](../notebooks/README.md) of this repository.

## Duckietown.
### Simulation Dataset
```
python build_duckietown_sim_data.py --dataset_dir duckietown --output_dir duckietown/sim/tfrecords
```
### Real Dataset:
```
python build_duckietown_sim_data.py --dataset_dir duckietown --output_dir duckietown/real/tfrecords
```
## Cityscapes
The folder structure is assumed to be:
  + datasets
    - build_cityscapes_data.py
    - convert_cityscapes.sh
    + cityscapes
      + cityscapesscripts (downloaded scripts: https://github.com/mcordts/cityscapesScripts)
      + gtFine
      + leftImg8bit
``` 
bash convert_cityscapes.sh
```
