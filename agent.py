"""
This is the agent.
"""
import environment

import time
import pandas as pd
import argparse
import keras
import numpy as np


class Solver:
    """
    This is an agent class with methods that can be used to interact with
    the Rubik's cube environment in environment.py
    """

    def __init__(self, state, network):
        """
        Initialize attributes for class
        :param state: Should be before any movement is done
        """
        # TODO make end_state a solution that is all the same color
        self.end_state = state.cube
        self.action_space = state.action_space
        self.network = network

    def action(self, cube, pretraining=False):
        """
        Perform an action on the rubik's cube environment according to a given
        policy
        :param cube:
        :param pretraining:
        :return:
        """
        if pretraining is True:
            return np.random.choice(self.action_space.flatten(), size=1)
        else:
            return self.network.predict(cube)

    def autodidactic(self, state):
        pass

    def reward(self, state=None):
        """
        Should calculate reward based on some of this information.
        Perhaps the number of moves it is away from win state
        :param state:
        :param action:
        :param next_state:
        :return:
        """
        reward_check = [len(np.unique(element)) for element in state]
        if len(set(reward_check)) == 1:
            return 1
        else:
            return 0


if __name__ == '__main__':
    cube = environment.Cube()
    agent = Solver(cube)

    cube.scramble_cube(3)

    agent.reward(cube.cube)









