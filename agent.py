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

    def __init__(self, state):
        """
        Initialize attributes for class
        :param state: Should be before any movement is done
        """
        # TODO make end_state a solution that is all the same color
        self.end_state = state.cube
        self.action_space = state.action_space


    def action(self, cube, pretraining=False):
        """
        Perform an action on the rubik's cube environment according to a given
        policy
        :param cube:
        :param pretraining:
        :return:
        """
        if pretraining is True:
            return np.random.choice(self.action_space, size=1)

        else:
            return keras.Model.predict()


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

        if (len(set(element)) == 1 for element in state):
            return 1
        else:
            return -1








