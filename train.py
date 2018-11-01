# import keras
# import tensorflow
import pandas as pd
import argparse
import configparser
from sklearn.model_selection import train_test_split
from data import read_data_set

from environment import Cube
from agent import Solver


config = configparser.ConfigParser()
config.read("config.ini")

# TODO Possible rewardsystem: 0 if nothing is achieved. 1 if progress is made (checkpoint)
# TODO                        -1 if not progress was made within a certain amount of moves.
# TODO                        Perhaps penalty is a bad solution to this, since it might be close
# TODO                        to a solution before it was penalized.


def parse_arguments():
    """
    Parse commandline arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    # Set up arguments

    parser.add_argument('-r', '--read_data', type=str, default=None,
                        help='Take in a .pkl file containing the data set')
    parser.add_argument('-w', '--generate_data', default=False, action='store_true',
                        help='If a filename is chosen, a data_set will be created')

    arguments = parser.parse_args()
    return arguments



class Network:
    """
    Follow the algorithm State, Action, Reward, State'

    Update should happen something like this:
        - Perform action
        - Check Reward
        - Go to new state
        - Store all of the above for the state before the action.
        - When it is time to train, use this information.

    We also need to see how the autodidactic algorithm works into this, since the number of states
    is about 3.6 million, it might use a really long time to train on this.
    """
    def __init__(self):

        self.training_samples = self.scramble_cube()

    def model_reinforcement(self):
        pass

    def model_supervised(self):
        pass

    def initialize_networks(self):
        pass

    def memory(self):
        pass

    def train(self):
        pass

    def remember(self):
        pass


    def scramble_cube(self,state, k):
        """
        Takes in a cube array, and scramble it k times.
        :param state
        :param k:
        :return:
        """


def split_data(data):
    """
    Splits the data into training, test and validation using sklearn lib.
    :param data: The data set
    :return:
    """
    return train_test_split(data["Cube"],
                            data["Actions"],
                            test_size=0.33)


def train(data, agent, cube):
    """
    Trains the data either according to supervised learning or reinforcement learning

    We (meaning Torstein) have not decided yet....
    :param data:
    :param agent:
    :param cube:
    :return:
    """

    x_train, x_test, y_train, y_test = split_data(data)  # Split data into training and test sets
    num_of_sim = config['model'].getint('num_of_sim')  # Fetch the amount of games

    # Start looping through simulations
    for simulation in range(num_of_sim):
        # Reset cube before each simulation
        cube.reset_cube()
        for cube, actions in zip(x_train, y_train):  # We don't use action at the moment

            # Predict an action
            agent.action(cube)
            reward = agent.reward(state=cube)





if __name__ == "__main__":

    # Parse arguments
    # args = parse_arguments()

    # Set up environment
    rubiks_cube = Cube()
    agent = Solver(rubiks_cube.cube)

    # Read data
    data, data_cube, data_actions = read_data_set(config['dataset']['read_file_name'])




    # Start training
    train(data, agent, rubiks_cube)




















