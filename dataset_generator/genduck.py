#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import argparse
import sys
import yaml
import random

import gym
from pyglet import app, clock
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.utils import get_file_path

parser = argparse.ArgumentParser()
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--style", default="photos", choices=["photos", "synthetic"])
args = parser.parse_args()
print(args)


def _load_map_mp(self, map_name: str):
        """
        Load the map layout from a YAML file
        """

        # Store the map name
        self.map_name = map_name

        # Get the full map file path
        self.map_file_path = get_file_path("maps", map_name, "yaml")

        with open(self.map_file_path, "r") as f:
            self.map_data = yaml.load(f, Loader=yaml.Loader)
            self.map_data['objects'][0]["rotate"] = random.randint(0, 360)
            self.map_data['objects'][0]["pos"] = (random.uniform(2.4,2.6), random.uniform(2,5))

        self._interpret_map(self.map_data)

Simulator._load_map = _load_map_mp


env = DuckietownEnv(
    map_name="generate_duckies",
    distortion=args.distortion,
    style=args.style,
)

env.reset()
env.render()

assert isinstance(env.unwrapped, Simulator)


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env.reset()
        env.render()
        return
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    elif symbol == key.P:
        env.reset()
        print("env reset")

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage depencency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     try:
    #         from experiments.utils import save_img
    #         save_img('screenshot.png', img)
    #     except BaseException as e:
    #         print(str(e))


def update(dt):
    env.render("free_cam")


# Main event loop
clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
app.run()

env.close()
