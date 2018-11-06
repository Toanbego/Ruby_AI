import keras
import argparse
import configparser
from sklearn.model_selection import train_test_split
import numpy as np
from collections import deque
import random
import time
import h5py

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
        self.eta = 0.0001  # Learning rate
        self.gamma = 0.95  # Discount rate
        self.decay = 0.0001
        self.batch_size = config['model'].getint('batch_size')
        self.memory = deque(maxlen=10000)
        self.pretraining = config['model'].getboolean('pretraining')

        # Initialize network
        self.network = self.model_reinforcement()

    def model_reinforcement(self, input=None):
        """
        Creates a neural network that takes in a flatted 6x2x2 array and returns
        the expected reward for a set of actions.
        :param input:
        :return:
        """

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1024, activation='relu',
                                     batch_size=self.batch_size,
                                     input_dim=24))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))

        model.add(keras.layers.Dense(12, activation='softmax'))

        adam = keras.optimizers.Adam(lr=self.eta, decay=self.decay)
        model.compile(optimizer=adam,
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def go_to_gym(self):
        """
        Method that trains the network with batches of data
        :return:
        """
        if self.pretraining is True:
            return
        # Sample a batch of data from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Append the state and target from the data
        x_batch, y_batch = [], []
        for state, act, target, next_state in minibatch:
            x_batch.append(state), y_batch.append(target)

        # Reshape the data to match the batch size
        states = np.array(x_batch).reshape(self.batch_size, 24)
        rewards = np.array(y_batch).reshape(self.batch_size, 12)

        self.network.fit(x=states, y=rewards,
                         epochs=1,
                         verbose=0,
                         batch_size=self.batch_size)


    def train(self, agent, cube):
        """
        Trains the data either according to supervised learning or reinforcement learning
        :param agent:
        :param cube:
        :return:
        """

        # x_train, x_test, y_train, y_test = split_data(data)  # Split data into training and test sets
        num_of_sim = config['model'].getint('num_of_sim')  # Fetch the amount of games
        max_steps = config['model'].getint('num_of_total_moves')  # Max amount of moves before "losing"

        solved_rate = deque(maxlen=40)
        solved_final = 0
        difficulty_level = 1

        # Start looping through simulations
        for simulation in range(1, num_of_sim):
            # Reset cube before each simulation
            cube.cube, cube.face = cube.reset_cube()

            # Scramble the cube as many times as the scramble_limit
            cube.cube = cube.scramble_cube(difficulty_level)[0]

            # After 10 simulations, pretraining is turned off
            if simulation > self.batch_size*5:
                self.pretraining = False

            for step in range(max_steps):
                # Get the state of the cube
                state = cube.cube

                # Get an action from agent and execute it
                act = agent.action(state.reshape(1, 24), self.pretraining)
                cube.rotate_cube(np.argmax(act))

                # Calculate reward and find the next state
                next_state = state
                reward = agent.reward(next_state)

                # TODO Everything after reward is an attempt to add the future reward
                target = reward #+ self.gamma * np.max(agent.action(next_state.reshape(1, 24), self.pretraining))
                target_vector = act
                target_vector[0][np.argmax(act)] = target

                # Append the result into the dataset
                self.memory.appendleft((state.reshape(1, 24), act, target_vector, next_state))

                # Go train, if pretraining is finished
                if self.pretraining is False:
                    self.go_to_gym()

                # Is the cube solved?
                if reward == 1:
                    # Solved score from the last 80% of the simulation
                    if simulation > simulation*0.8:
                        solved_final += 1
                    solved_rate.appendleft(1)
                    break

            # If the reward is zero here, it means the cube was not solved
            if reward == 0:
                solved_rate.appendleft(0)

            # Calculate score
            if simulation % 10 == 0:
                solved = (sum(solved_rate) / len(solved_rate))
                print(f"\033[93m"
                      f"{sum(solved_rate)} Cubes solved of the last 40, accuracy: {round(solved, 2)}"
                      f"\033[0m")

                # Saves the model if the model is deemed good enough
                if round(solved, 2) == 1:
                    print("Increasing the number of scrambles by 1")
                    difficulty_level += 1
                    keras.models.save_model(self.network, f"models/solves_one_scramble - {time.time()}.h5")

        print(f"Final difficulty level: {difficulty_level}")
        print(solved_final/(simulation*0.8))
        print(solved_final)
        print((simulation*0.8))

    def test(self):
        """
        Solves a cube with a trained model. Always renders an image
        :return:
        """
        pass


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
    Main function of the script
    :return:
    """
    # Set up environment
    rubiks_cube = Cube()

    # Initialize network
    model = Network(rubiks_cube.cube)

    # Set up agent
    agent = Solver(rubiks_cube, model.network)

    # Start training
    if config['model'].getboolean('train'):
        model.train(agent, rubiks_cube)

    elif config['model'].getboolean('test'):
        model.test(agent, rubiks_cube)


if __name__ == '__main__':
    main()

















