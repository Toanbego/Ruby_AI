"""
This is the agent.
"""
import environment
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
        self.end_state = state.cube
        self.action_space = state.action_space


    def action(self, cube, network):
        """
        Perform an action on the rubik's cube environment according to a given
        policy
        :param cube:
        :param pretraining:
        :return:
        """
        return network.predict(cube)


    def reward(self, state=None):
        """
        Should calculate reward based on some of this information.
        Perhaps the number of moves it is away from win state
        :param state:
        :return:
        """
        reward_check = [len(np.unique(element)) for element in state]
        if sum(reward_check) == 6:
            return 1
        else:
            return 0


if __name__ == '__main__':
    cube = environment.Cube()
    agent = Solver(cube)

    cube.scramble_cube(3)

    agent.reward(cube.cube)









