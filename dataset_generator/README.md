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

## Generating a dataset

Open *dataset_generator_notebook.ipynb* in Colab

Modify the line 
```
python3 dataset_generator.py 1 10
```
according to this syntax
```
python3 dataset_generator.py seed number_of_samples
```

The dataset will be saved in your Google Drive in a folder named *images*

:warning: You will be sked to give Colab your permission to access Drive. This will happen at the end of dataset generation. It could be useful to give this permission before running the script.
