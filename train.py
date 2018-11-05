import tensorflow as tf
import keras
import pandas as pd
import argparse
import configparser
from sklearn.model_selection import train_test_split
import gym
import numpy as np
from collections import deque
import random

from environment import Cube
from agent import Solver
from data import read_data_set

config = configparser.ConfigParser()
config.read("config.ini")



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
    def __init__(self, cube):
        self.cube = cube
        self.eta = 0.001  # Learning rate
        self.gamma = 0.95  # Discount rate
        self.decay = 0.0001
        self.batch_size = config['model'].getint('batch_size')
        self.memory = deque(maxlen=10000)

        # Initialize network
        self.network = self.model_reinforcement()

    def model_reinforcement(self, input=None):
        """
            Creates a neural network that takes in a 6x2x2 array and returns
            the expected reward for a set of actions.
            :param input:
            :return:
            """

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(4096, activation='relu', input_dim=24))
        model.add(keras.layers.Dense(2048, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dense(12, activation='softmax'))

        adam = keras.optimizers.Adam(lr=self.eta, decay=self.decay)
        model.compile(optimizer=adam,
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def go_to_gym(self, states, rewards):
        if len(self.memory) < 30:
            return

        # minibatch = random.sample(self.memory, self.batch_size)
        # #TODO make this a one line for loop or no loop at all
        # x_batch, y_batch = [], []
        # for state, act, target, next_state in minibatch:
        #     x_batch.append(state), y_batch.append(target)

        # states = np.array(list(zip(minibatch))[0])
        # rewards = np.array(list(zip(minibatch))[2])

        self.network.fit(x=states, y=rewards,
                         epochs=1,
                         batch_size=1,
                         verbose=0)


    def train(self, data, agent, cube):
        """
        Trains the data either according to supervised learning or reinforcement learning
        :param data:
        :param agent:
        :param cube:
        :return:
        """

        # x_train, x_test, y_train, y_test = split_data(data)  # Split data into training and test sets
        num_of_sim = config['model'].getint('num_of_sim')  # Fetch the amount of games
        max_steps = config['model'].getint('num_of_total_moves')
        pretraining = config['model'].getboolean('pretraining')

        solved_rate = deque(maxlen=40)

        # Start looping through simulations
        for simulation in range(1, num_of_sim):
            # Reset cube before each simulation
            cube.cube, cube.face = cube.reset_cube()
            cube.cube = cube.scramble_cube(1)[0]

            # After 10 simulations, pretraining is turned off
            if simulation > 10:
                pretraining = False

            for step in range(max_steps):
                # Get the state of the cube
                state = cube.cube

                # Get an action from agent and execute it
                act = agent.action(state.reshape(1, 24), pretraining)

                cube.rotate_cube(np.argmax(act))

                # Calculate reward and find the next state
                reward = agent.reward(cube.cube)
                next_state = state

                #TODO legg til discount rate ved å prøve å predict next_state også
                target = reward #+ self.gamma * np.max(agent.action(next_state.reshape(1, 24), pretraining))
                target_vector = act
                target_vector[0][np.argmax(act)] = target

                # Append the result into the dataset
                self.memory.appendleft((state.reshape(1, 24), act, target_vector, next_state))

                # Is the cube solved?
                if reward == 1:
                    solved_rate.appendleft(1)
                    print("\033[93mcube is solved\033[0m")
                    break

                # Go train
                self.go_to_gym(state.reshape((1, 24)), target_vector)

            # If the reward is zero here, it means the cube was not solved
            if reward == 0:
                solved_rate.appendleft(0)
            # Calculate score
            if simulation % 10 == 0:
                solved = sum(solved_rate) / len(solved_rate)
                print(f'solved rate for last 40 cubes: {solved.round(2)}')
                
                # # Check if over 50% is solved
                # if sum(solved_rate) > len(solved_rate)/2:
                #     solved = sum(solved_rate)/len(solved_rate)
                #     print(f'solved rate for last 40 cubes: {solved}')



def split_data(data):
    """
    Splits the data into training, test and validation using sklearn lib.
    :param data: The data set
    :return:
    """
    return train_test_split(data["Cube"],
                            data["Actions"],
                            test_size=0.33)


def main():
    """

    :return:
    """

    # Parse arguments
    # args = parse_arguments()

    # Read data
    if config['dataset'].getboolean('read_data') is True:
        data, data_cube, data_actions = read_data_set(config['dataset']['read_file_name'])

        # Start training
        # Initialize network
        # Set up environment
        rubiks_cube = Cube()
        model = Network(rubiks_cube.cube)
        agent = Solver(rubiks_cube, model.network)
        model.train(data, agent, rubiks_cube)


if __name__ == '__main__':

    main()

















