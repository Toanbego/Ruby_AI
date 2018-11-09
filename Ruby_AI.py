import keras
import argparse
import configparser
import numpy as np
from collections import deque
import random
import time
import math
import copy
# import h5py

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
        self.eta = config['network'].getfloat('learning_rate')  # Learning rate
        self.gamma = 0.95  # Discount rate
        self.decay = 0.995
        self.epsilon = config['network'].getfloat('epsilon')
        self.epsilon_min = config['network'].getfloat('epsilon_min')
        self.epsilon_decay = config['network'].getfloat('epsilon_decay')
        self.batch_size = config['network'].getint('batch_size')
        self.memory = deque(maxlen=5000)
        self.pretraining = config['simulation'].getboolean('pretraining')
        self.difficulty_level = 1

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
        # model.add(keras.utils.normalize(input))
        model.add(keras.layers.Dense(1024, activation='relu',
                                     batch_size=self.batch_size))
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

        # Create a mini batch from memory
        states, rewards = self.sample_memory()

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
        # TODO Exploration er ett problem ser det ut til.
        # TODO prioritized replay. Større sjangse for at en state med rewward blir valgt

        num_of_sim = config['simulation'].getint('num_of_sim')  # Fetch the amount of games
        max_steps = config['simulation'].getint('num_of_total_moves')  # Max amount of moves before "losing"

        solved_rate = deque(maxlen=40)
        solved_final = 0
        self.best_accuracy = 0.0
        self.difficulty_level = 1
        self.epsilon_decay_steps = 0

        # Start looping through simulations
        simulation = 0
        while self.difficulty_level < 3:
        # for simulation in range(1, num_of_sim):
            # Reset cube before each simulation
            cube.cube, cube.face = cube.reset_cube()

            # Scramble the cube as many times as the scramble_limit
            cube.scramble_cube(self.difficulty_level)
            # cube.rotate_cube('F')

            # After 10 simulations, pretraining is turned off
            if simulation > 1000:
                self.pretraining = False

            for step in range(self.difficulty_level):
                # Get the state of the cube
                state = copy.deepcopy(cube.cube)

                # Get an action from agent and execute it
                actions = agent.action(state.reshape(1, 24), self.pretraining)

                # Choose predicted action or explore.
                if self.pretraining is True or self.get_epsilon(self.epsilon_decay_steps) >= np.random.random():
                    self.epsilon_decay_steps += 1
                    take_action = np.random.random_integers(0, 11, 1)
                else:
                    take_action = np.argmax(actions)

                # if not self.pretraining:
                #     if cube.num_to_str[int(take_action)] == 'Fr' or cube.num_to_str[int(take_action)] == 'Br':
                #         s = cube.num_to_str[int(take_action)]
                #         print(s)
                # Execute action
                next_state, face = cube.rotate_cube(take_action)

                # Calculate reward and find the next state
                reward = agent.reward(next_state)

                # TODO Everything after reward is an attempt to add the future reward
                target_vector = self.create_target_vector(reward, actions, take_action)

                # Append the result into the dataset
                self.memory.appendleft((state.reshape(1, 24), actions, target_vector, next_state))

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

            simulation += 1
            # self.epsilon *= self.epsilon_decay
            # If the reward is zero here, it means the cube was not solved
            if reward == 0:
                solved_rate.appendleft(0)
            if self.pretraining is False:
                self.check_progress(simulation, solved_rate)
            else:
                self.check_progress(simulation, solved_rate)


        print(f"Final difficulty level: {self.difficulty_level}")
        print(f" The best accuracy: {self.best_accuracy}")

    def create_target_vector(self, reward, actions, take_action):
        """
        Creates the target vector used as label in the training.
        Will also look at future rewards with a discount factor
        :param reward:
        :param actions:
        :return:
        """
        if self.difficulty_level > 1:
            target = reward  # + self.gamma * np.max(agent.action(next_state.reshape(1, 24), self.pretraining))
        else:
            target = reward
        target_vector = actions.copy()
        target_vector[0][take_action] = target
        return target_vector

    def sample_memory(self):
        """
        Creates a mini batch from the memory
        Currently uniform selection is the only option. Should apply
        prioritized selection as well
        :return:
        """
        # Sample a batch of data from memory
        mini_batch = random.sample(self.memory, self.batch_size)

        # selection_wheel = [(fitness / sum(self.memory[2])) for fitness in self.memory[2]]
        # mini_batch = np.random.choice(self.memory, p=selection_wheel)  # Random selection

        # Append the state and target from the data
        x_batch, y_batch = [], []
        for state, act, target, next_state in mini_batch:
            x_batch.append(state), y_batch.append(target)

        # Reshape the data to match the batch size
        states = np.array(x_batch).reshape(self.batch_size, 24)
        rewards = np.array(y_batch).reshape(self.batch_size, 12)
        return states, rewards

    def get_epsilon(self, decay_steps):
        """
        Returns the e - greedy
        :param simulation:
        :return:
        """
        # return max(self.epsilon_min, self.epsilon)
        if self.pretraining is False:
            return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((decay_steps + 1) * self.epsilon_decay)))
        else:
            return 1

    def test(self):
        """
        Solves a cube with a trained model. Always renders an image
        :return:
        """
        pass

    def check_progress(self, simulation, solved_rate):
        """
        Checks the progress of the training and increases the number of scrambles if it deems
        the network good enough
        :param difficulty_level:
        :param simulation:
        :param solved_rate:
        :return:
        """
        # Calculate score
        if simulation % len(solved_rate) == 0:
            solved = (sum(solved_rate) / len(solved_rate))
            print(f"\033[93m"
                  f"{sum(solved_rate)} Cubes solved of the last {len(solved_rate)} \naccuracy: {round(solved, 2)}"
                  f"\nExploration rate: {self.get_epsilon(self.epsilon_decay_steps)}\nScrambles {self.difficulty_level}"
                  f"\n================================="
                  f"\033[0m")
            if solved > self.best_accuracy:
                self.best_accuracy = solved

            # Saves the model if the model is deemed good enough
            if round(solved, 2) == 1 and self.pretraining is False:
                self.best_accuracy = 0
                self.epsilon_decay_steps = 0
                self.epsilon = 1
                print("Increasing the number of scrambles by 1")
                self.difficulty_level += 1
                keras.models.save_model(self.network, f"models/solves_{self.difficulty_level}_scrambles - {time.time()}.h5")


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
    if config['simulation'].getboolean('train'):
        model.train(agent, rubiks_cube)


    elif config['simulation'].getboolean('test'):
        model.test(agent, rubiks_cube)


if __name__ == '__main__':
    main()

















