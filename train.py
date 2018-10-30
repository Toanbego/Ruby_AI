# import keras
# import tensorflow
import pandas as pd
import argparse
import configparser
config = configparser.ConfigParser()
config.read("config.ini")

print(config['model']["num_of_sim"])



from environment import Cube
from agent import Solver

# TODO Consider making a main class for models, with subclasses for reinforcement and supervised

# TODO Since we have 3.600.000 million combinations, perhaps it is possible to completely use supervise learning
# TODO A possible reward could be how many moves we are away from the solution.


def parse_arguments():
    """
    Parse commandline arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    # Set up arguments

    parser.add_argument('-r', '--read_data', type=str, default=None,
                        help='Take in a .pkl file containing the data set')
    parser.add_argument('-g', '--generate_data', type=str, default=None,
                        help='If a filename is chosen, a data_set will be created')

    arguments = parser.parse_args()
    return arguments


def generate_training_samples(data_set_size, cube, file_name, scrambles=1):
    """
    Generate the training set for supervised learning
    :param data_set_size: size of data set
    :param scrambles: How many times to maximum scramble a cube
    :param cube: Cube object
    :param file_name:
    :return:
    """

    data_set = []   # Will consist of the cube array

    # Generate randomly scrambled cubes
    for sample in range(data_set_size):

        # Reset the cube each time
        cube.cube, cube.face = cube.reset_cube()

        # Append the scrambled cube, with the actions it took to get there
        data_set.append(cube.scramble_cube(scrambles))

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


# def train(data, agent, cube):
#     num_of_sim = config.ini[model][number_of_simulations]


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # Set up environment
    rubiks_cube = Cube()

    # Generate Training set
    if args.generate_data is not None:
        generate_training_samples(1000, rubiks_cube, args.generate_data, scrambles=2)
    elif args.read_data is not None:
        data, data_cube, data_actions = read_data_set(args.read_data)
        print(data_cube.head()[0])
    # agent = Solver(rubiks_cube.cube)
    #
    # reward = agent.reward(state=rubiks_cube.cube)

    # agent.action('F', 1)
    # agent.action('F', -1)
    # # Calculate reward
    # reward = agent.reward(rubiks_cube.cube)
    # print(reward)



















