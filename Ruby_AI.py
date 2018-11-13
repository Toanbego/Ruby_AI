import keras
import argparse
import configparser
import numpy as np
import itertools
from collections import deque
import collections
import random
import time
import math
import copy
import re

# import h5py

from environment import Cube
from agent import Solver


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
        self.gamma = config['network'].getfloat('discount_rate')  # Discount rate
        self.decay = 0.995
        self.epsilon = config['network'].getfloat('epsilon')
        self.epsilon_min = config['network'].getfloat('epsilon_min')
        self.epsilon_decay = config['network'].getfloat('epsilon_decay')
        self.batch_size = config['network'].getint('batch_size')
        self.memory = deque(maxlen=1001)
        self.pretraining = config['simulation'].getboolean('pretraining')
        self.epoch = config['model'].getint('epoch')

        self.load_weights = config['network'].getboolean('load_weights')
        self.load_model_path = config['network']['load_model']

        self.difficulty_level = 1
        self.best_accuracy = 0.0
        self.difficulty_counter = 0

        self.tensorboard = keras.callbacks.TensorBoard(log_dir="logs/", update_freq=200)

        # Initialize network
        if self.load_weights is True:
            self.network = self.load_network()
        else:
            self.network = self.model_reinforcement()


    def model_reinforcement(self, input=None):
        """
        Creates a neural network that takes in a flatted 6x2x2 array and returns
        the expected reward for a set of actions.
        :param input:
        :return:
        """

        model = keras.models.Sequential()

        model.add(keras.layers.Dense(512, activation='relu',
                                     batch_size=self.batch_size))
        # model.add(keras.layers.Dropout(0.05))

        model.add(keras.layers.Dense(256, activation='relu'))
        # model.add(keras.layers.Dropout(0.05))

        model.add(keras.layers.Dense(128, activation='relu'))

        model.add(keras.layers.Dense(12, activation='softmax'))

        model.compile(loss=keras.losses.mse,
                      lr=self.eta,
                      # decay=self.decay,
                      optimizer=keras.optimizers.adadelta(),
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

    def load_network(self):
        """
        Loads a pretrained network
        :return:
        """
        return keras.models.load_model(self.load_model_path)

    def train(self, agent, cube):
        """
        Trains the data either according to supervised learning or reinforcement learning
        :param agent:
        :param cube:
        :return:
        """

        num_of_sim = config['simulation'].getint('num_of_sim')  # Fetch the amount of games
        max_steps = config['simulation'].getint('num_of_total_moves')  # Max amount of moves before "losing"

        solved_rate = deque(maxlen=40)
        solved_final = 0

        # Check the last difficulty level if loading weights
        if self.load_weights is True:
            self.difficulty_level = int(re.search('_(\d)_', self.load_model_path)[0].strip('_'))
        else:
            self.difficulty_level = 1

        self.epsilon_decay_steps = 0

        # Start looping through simulations
        simulation = 0
        try:
            while self.difficulty_level:

            # for simulation in range(1, num_of_sim):
                    # Reset cube before each simulation
                    cube.cube, cube.face = cube.reset_cube()

                    # Scramble the cube as many times as the scramble_limit
                    _, self.scramble_actions = cube.scramble_cube(self.difficulty_level)

                    # After 1000 simulations, pretraining is turned off
                    if len(self.memory) >= 1000:
                        self.pretraining = False

                    for step in range(self.difficulty_level):

                        # Get the state of the cube
                        state = copy.deepcopy(cube.cube)
                        # state = keras.utils.normalize(copy.deepcopy(cube.cube), order=2)

                        # Take in state and predict the reward for all possible actions
                        actions = agent.action(state.reshape(1, 24), self.network)

                        # Either choose predicted action or explore depending on current epsilon value
                        if self.pretraining is True or self.get_epsilon(self.epsilon_decay_steps) >= np.random.random():
                            self.epsilon_decay_steps += 1
                            take_action = np.random.random_integers(0, 11, 1)
                        else:
                            take_action = np.argmax(actions)

                        # Execute action
                        next_state, face = cube.rotate_cube(take_action)

                        # Calculate reward and find the next state
                        reward = agent.reward(next_state)

                        # TODO Everything after reward is an attempt to add the future reward
                        target_vector = self.create_target_vector(agent, next_state, reward, actions, take_action, step, cube)

                        # Append the result into the dataset
                        # if step > 0 and np.random.random() > 0.8:
                        #     self.memory.appendleft((state.reshape(1, 24), actions, target_vector, next_state))
                        # elif step == 0:
                        self.memory.appendleft((state.reshape(1, 24), actions, target_vector, next_state))

                        for train in range(15):
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

                    # Increase the simulation counter
                    simulation += 1

                    # If the reward is zero here, it means the cube was not solved
                    if reward == 0:
                        solved_rate.appendleft(0)
                    if self.pretraining is False:
                        self.check_progress(simulation, solved_rate)
                    else:
                        self.check_progress(simulation, solved_rate)

        except KeyboardInterrupt:
            print("Stop training and save model")
        finally:
            print("i was here")




        print(f"Final difficulty level: {self.difficulty_level}")
        print(f" The best accuracy: {self.best_accuracy}")

    def create_target_vector(self, agent, next_state, reward, actions, take_action, step, cube):
        """
        Creates the target vector used as label in the training.
        Will also look at future rewards with a discount factor
        :param reward:
        :param actions:
        :return:
        """

        check_future = agent.action(next_state.reshape(1, 24), self.network)

        if np.max(check_future) > 0.9:
            target = reward + self.gamma*np.argmax(check_future)

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
        # mini_batch = collections.deque(itertools.islice(self.memory, 0, self.batch_size))

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
                  f"\nExploration rate: {round(self.get_epsilon(self.epsilon_decay_steps), 5)}"
                  f"\nScrambles {self.difficulty_level}"
                  f"\nBest accuracy: {self.best_accuracy}, reached 100% {self.difficulty_counter} times"
                  f"\n================================="
                  f"\033[0m")
            if solved > self.best_accuracy:
                self.best_accuracy = solved

            # Saves the model if the model is deemed good enough
            if round(solved, 2) == 1 and self.pretraining is False:
                self.difficulty_counter += 1
                if self.difficulty_counter > 7:

                    # Reset variables
                    self.difficulty_counter = 0
                    self.best_accuracy = 0
                    self.epsilon_decay_steps = 0

                    # Increment d
                    self.epsilon = config['network'].getfloat('epsilon')
                    print("Increasing the number of scrambles by 1")
                    self.difficulty_level += 1
                    if config['simulation'].getboolean('test') is not True:
                        keras.models.save_model(self.network, f"models/solves_{self.difficulty_level}_scrambles - {time.time()}.h5")

    def test(self, agent, cube):
        """
        Method for testing a trained model
        :param agent:
        :param cube:
        :return:
        """
        self.network = self.load_network()
        solved_rate = deque(maxlen=40)
        solved_final = 0
        self.best_accuracy = 0.0
        self.difficulty_counter = 0
        self.difficulty_level = 1
        self.epsilon_decay_steps = 0
        self.action_stat = np.zeros(12, dtype=int)
        # Start looping through simulations
        simulation = 0
        while self.difficulty_level < 3:

            try:
                # Reset cube before each simulation
                cube.cube, cube.face = cube.reset_cube()

                # Scramble the cube as many times as the scramble_limit
                _, scramble_actions = cube.scramble_cube(self.difficulty_level, render_image=True)

                for step in range(self.difficulty_level):

                    # Get the state of the cube
                    state = copy.deepcopy(cube.cube)
                    # state = keras.utils.normalize(cube.cube, order=2)

                    # Get an action from agent and execute it
                    actions = agent.action(state.reshape(1, 24), self.network)

                    # Choose predicted action or explore.
                    take_action = np.argmax(actions)

                    # Execute action
                    next_state, face = cube.rotate_cube(take_action, render_image=True)

                    # Calculate reward and find the next state
                    reward = agent.reward(next_state)

                    target_vector = self.create_target_vector(agent, next_state, reward, actions, take_action)

                    # Statistics
                    self.action_stat[take_action] += 1
                    if simulation % 100 == 0:
                        print(self.action_stat)
                        self.action_stat = np.zeros(12, dtype=int)

                    # Is the cube solved?
                    if reward == 1:

                        # Solved score from the last 80% of the simulation
                        if simulation > simulation * 0.8:
                            solved_final += 1
                        solved_rate.appendleft(1)
                        break

                # Increase the simulation counter
                simulation += 1

                # If the reward is zero here, it means the cube was not solved
                if reward == 0:
                    solved_rate.appendleft(0)
                if self.pretraining is False:
                    self.check_progress(simulation, solved_rate)
                else:
                    self.check_progress(simulation, solved_rate)

            except Exception as e:
                print(e)
                print("i was here")
                keras.models.save_model(self.network,
                                        f"models/TEST{self.difficulty_level}_scrambles - {time.time()}.h5")
                break

        print(f"Final difficulty level: {self.difficulty_level}")
        print(f" The best accuracy: {self.best_accuracy}")


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
    agent = Solver(rubiks_cube)

    # Start training
    if config['simulation'].getboolean('train'):
        model.train(agent, rubiks_cube)

    elif config['simulation'].getboolean('test'):
        model.test(agent, rubiks_cube)


if __name__ == '__main__':
    main()

















