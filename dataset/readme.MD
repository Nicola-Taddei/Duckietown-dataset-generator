# How to prepare datasets.
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
