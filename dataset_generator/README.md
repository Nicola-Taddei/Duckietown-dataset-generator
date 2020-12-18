# Gym-Duckietown Dataset Generator
This is the dataset generator for the mobile segmentation project for Duckietown. It can output the following:
- Simulated RGB images
- Semantic Segmentation of the Simulated RGB Images
- Segmention Maps of the Bezier Lines of the middle of the lanes.
- Semantic Segmentation + Bezier Lines of the middle of the lanes.

## Introduction
This part of the repository is a fork from the Gym-Duckietown is a simulator for the [Duckietown](https://duckietown.org) Universe, written in pure Python/OpenGL (Pyglet). 



# Getting Started

## Requirements

Requirements:
- Python 3.6+
- OpenAI gym
- NumPy
- Pyglet
- PyYAML
- cloudpickle
- PyTorch


## Installation Using PIP

You can install all the dependencies except PyTorch with `pip3`:

It might be a good idea to create a new conda environment first.
```
conda create -n duckietown_gym_py38 python=3.8
conda activate duckietown_gym_py38
```
Then
```
pip3 install -e .
```

## Genrating a dataset

It was used for generating the semantic segmentation dataset for this project. The script 
`dataset_generator.py` was adapted from `manual_control.py`, and used for generating a 10,000 image semantic segmentation
dataset, using the following command: 

`python3 dataset_generator.py --env-name 'Duckietown-udem1-v0' --map-name 'loop_empty' --dataset-size 10000 --resize 2`

Once in the simulator, the following keys can be used to launch the following effects: 
  - P: Reset the environment
  - L: Print the lane pose information
  - N: Turn left
  - M: Turn Right
  - R: Extract image and semantic segmenation annotations for observation
  - A: Automatically extract the desired number of images (--dataset-size) by reseting the environment and extracting the annotations each time.

The dataset appears in the folder **./datasets/image_XXXXX**, with the following folder structure: 

```
image_XXXX_XX_XX_XX_XX_XX
  ├──'bezier_only'				
  |      ├──labels				  
  |      |   ├──0.npy			
  |      |   ├──1.npy			 
  |      |   └──...					 				 
  |      ├──rgb_orig				
  |      |   ├──0.png		
  |      |   ├──1.png			 
  |      |   └──...							    	
  |      └──rgb_ss			
  |          ├──0_seg.png		
  |          ├──1_seg.png			 
  |          └──...
  ├──'w_bezier'
  |      └──...	  
  └──'wo_bezier'
         └──...	  
```

The folder **bezier_only** contains the semantic segmentation annotations for only the left and right lane classes. 
The folder **w_bezier** contains the annotations for left and right bezier classes, as well as duckies, white lines
and yellow lines. The folder **wo_bezier** contains only the duckies, white lines and yellow lines. 

