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


## Troubleshooting

Here is some common problem your might encounter with our solution.

Symptom: The Duckiebot does not move.

Resolution: Check battery charge. Hard reboot the Duckybot. (I know, but it is still worth mentionning)

Symptom: Segmentation Quality is poor, and the Duckiebot crashes.

Resolution: We fine-tuned different models for different task. Please make sure than the model specified in the Object Detection Node is the right one. See the following two files:

- exercise_ws/src/object_detection/src/object_detection_node.py
- exercise_ws/src/object_detection/include/object_detection/model.py

Symptom: Segmentation is good, but the Duckiebot goes really fast in erratic motion.

Resolution: Change the speed, turn_speed, K and D parameters in the lane controller node until the Duckie becomes well-behaved.

Symptom: Duckiebot drives fine, but looses control and starts oscillating.

Resolution: Go slower! Lower "turn_speed" and "speed". A good idea is to start with low speeds (0.2 - 0.3) and then increase speed and gains (K, D) iteratively.

Symptom: Duckiebot oscillates even at slow speeds.

Resolution: Lower the K gain until it stop oscillating. Increase the "D" gain to get better performance in curves. Don't increase too much though, it will oscillate again at some point!

Symptom: Duckiebot has not fear and can crash into objects.

Resolution: Go slower! The perception system is limited by the network latency and CPU power of your machine. Running at 30 FPS on a Desktop allows going faster than running at 12 frames per seconds on a 3 years old laptop. A faster CPU will give the Duckiebot better reflexes. If you have time, do the inference on GPU, this would allow a laptop to process frames much faster, and give better reaction times to the Duckiebot.

## Demo failure demonstration

Here is a video of the Duckiebot going way too fast. It will eventually crash, just a matter of time!

https://vimeo.com/494207868

