# Pure-Pursuit Controller for the Duckietown Sementic Segmentation Project
This is based on the Object Detection Exercise repository.

## Getting Started
At this point, we assume that the duckietown setup was completed. 

## How to launch solution:
```
~/MILA/duckie_ift6757/mobile-segmentation/solution$ dts exercises test --sim
``` 

## How to setup dataloging
It is possible to log data
```
rosparam set /agent/datalog True
``` 
The logs will be in 
```
/code/exercises_ws/datalog
```
Those logs can be analysed with the [controller_exploration notebook](../notebooks/controller_exploration.ipynb)
## How to submit.
```
dts challenges submit --challenge aido5-LF-sim-validation
```
