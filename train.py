# import keras
# import tensorflow
import pandas as pd
import argparse
import configparser
from sklearn.model_selection import train_test_split
import itertools
from numba import jit

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


def generate_deterministic_data(cube, generate_deterministic):
    """
    Generates a deterministic data set
    :param cube:
    :param generate_deterministic: If true: continue, If false: return
    :return:
    """
    if generate_deterministic is False:
        return
    # Initialize variable
    nodes = config['dataset'].getint('nodes')
    file_name = config['dataset']['write_file_name']
    data_set = []
    list_of_permutations = []

    # Permute the number of possible combinations up to max 10 permutations
    noe = 0
    for node in range(1, nodes+1):
        list_of_permutations.append(list(itertools.permutations(cube.action_space, node)))
        noe += len(list_of_permutations[node-1])
    print("finished with permutations")
    # Make a search tree
    for permutations in list_of_permutations:  # Loop through action sequence
        for action_seq in permutations:
            cube.cube, cube.face = cube.reset_cube()  # Reset the cube each time
            for action in action_seq:
                cube.rotate_cube(action)  # Perform actions
            data_set.append((cube.cube, action_seq))

    # Create Data frame and write to .pkl
    df = pd.DataFrame(data_set, columns={"Cube": data_set[:][0], "Actions": data_set[:][1]})
    print("Writing dataset")
    df.to_pickle(f"data/{file_name}.pkl")


def generate_training_samples(cube, generate_random, scrambles=1):
    """
    Generate the training set for supervised learning
    :param cube:
    :param generate_random:
    :param scrambles:
    :return:
    """
    if generate_random is False:
        return
    data_set = []   # Will consist of the cube array
    data_set_size = config['dataset'].getint('sample_size')
    file_name = config['dataset']['write_file_name']
    print("Scrambling cubes")
    # Generate randomly scrambled cubes
    for sample in range(data_set_size):

        # Reset the cube each time
        cube.cube, cube.face = cube.reset_cube()

        # Append the scrambled cube, with the actions it took to get there
        data_set.append(cube.scramble_cube(scrambles))
    print("Writing to file")
    # Create Data frame and write to .pkl
    df = pd.DataFrame(data_set, columns={"Cube": data_set[:][0], "Actions": data_set[:][1]})
    df.to_pickle(f"data/{file_name}.pkl")


def read_data_set(file_name):
    """
    Reads a .pkl file an returns the columns separated into cubes
    and actions
    :param file_name:
    :return:
    """

    df = pd.read_pickle(f"data/{file_name}.pkl")
    return df, df["Cube"], df["Actions"]


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

    # Generate Training set if set to True
    generate_deterministic_data(rubiks_cube, config['dataset'].getboolean('generate_deterministic'))
    generate_training_samples(rubiks_cube, config['dataset'].getboolean('generate_random'), scrambles=2)

    # Read data
    data, data_cube, data_actions = read_data_set(config['dataset']['read_file_name'])




    # if args.generate_data is not None:
    #
    #     #
    # elif args.read_data is not None:
    #
    # print(data_cube, data_actions)

    # agent = Solver(rubiks_cube.cube)

    # Start training
    # train(data, agent, rubiks_cube)

    # reward = agent.reward(state=rubiks_cube.cube)

    # agent.action('F', 1)
    # agent.action('F', -1)
    # # Calculate reward
    # reward = agent.reward(rubiks_cube.cube)
    # print(reward)



















