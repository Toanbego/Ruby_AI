"""
This is the agent.
"""
import environment
from numba import jit
import time
import pandas as pd
import argparse


def parse_arguments():
    """
    Parse commandline arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    # Set up arguments

    parser.add_argument('--read_data', type=str, default=None,
                        help='Take in a .pkl file containing the data set')
    parser.add_argument('--generate_data', type=str, default=None,
                        help='If a filename is chosen, a data_set will be created')

    arguments = parser.parse_args()
    return arguments

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
        self.end_state = state
        # self.reward = self.reward(self.end_state)

    def action(self, face):
        """
        Perform an action on the rubik's cube environment according to a given
        policy
        :param face:
        :param dir:
        :return:
        """
        rubiks_cube.rotate_cube(face)

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
        print(state)
        if state.all() == self.end_state.all():
            return 1
        else:
            return -1


def generate_training_samples(data_set_size, scrambles, rubiks_cube, file_name):
    """
    Generate the training set for supervised learning
    :param data_set_size: size of data set
    :param scrambles: How many times to maximum scramble a cube
    :param rubiks_cube: Cube object
    :param file_name:
    :return:
    """
    # TODO It might also be an idea to not make the dataset randomly scrambled, but deterministic
    data_set = []   # Will consist of the cube array

    # Generate randomly scrambled cubes
    for sample in range(data_set_size):

        # Reset the cube each time
        rubiks_cube.cube, rubiks_cube.face = rubiks_cube.reset_cube()

        # Append the scrambled cube, with the actions it took to get there
        data_set.append(rubiks_cube.scramble_cube(scrambles))

    # Create Data frame and write to .pkl
    df = pd.DataFrame(data_set, columns={"Cube": data_set[:][0], "Actions": data_set[:][1]})
    df.to_pickle(f"{file_name}.pkl")


def read_data_set(file_name):
    """
    Reads a .pkl file an returns the columns separated into cubes
    and actions
    :param file_name:
    :return:
    """
    df = pd.read_pickle(f"{file_name}.pkl")
    return df, df["Cube"], df["Actions"]


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # Set up environment
    rubiks_cube = environment.Cube()

    # Generate Training set
    if args.generate_data is not None:
        generate_training_samples(100, 2, rubiks_cube, args.generate_data)
    elif args.read_data is not None:
        data, data_cube, data_actions = read_data_set(args.read_data)

    # Initialize agent
    agent = Solver(rubiks_cube.cube)

    # Testing some methods




    # agent.action('F', 1)
    # agent.action('F', -1)
    # # Calculate reward
    # reward = agent.reward(rubiks_cube.cube)
    # print(reward)





