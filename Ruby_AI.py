import keras
import configparser
import numpy as np
from collections import deque
import random
import time
import math
import copy
import re
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from environment import Cube
import sys


config = configparser.ConfigParser()
config.read("config.ini")


class Network:
    """
    Follow the algorithm State, Action, Reward, State'


    Attributes:
        train_model
        test_model
        input_shape
        choose_net
        eta
        gamma
        decay
        epsilon
        epsilon_min
        epsilon_decay
        batch_size
        pretraining
        epoch
        threshold
        one_hot
        evaluate
        number_of_cubes_to_solve
        self.done
        self.solved
        self.epsilon_decay_steps
        self.simulations_this_scrambles

        self.memory

        self.difficulty_level
        self.best_accuracy
        self.difficulty_counter
        self.accuracy_points
        self.axis
        self.plot_progress
        self.check_level
        self.difficulty_test
        self.number_of_moves_eval


    We also need to see how the autodidactic algorithm works into this, since the number of states
    is about 3.6 million, it might use a really long time to train on this.
    """
    def __init__(self, cube):
        self.cube = cube

        # Network parameters
        self.train_model = config['simulation'].getboolean('train')
        self.test_model = config['simulation'].getboolean('test')
        self.input_shape = tuple([int(s) for s in config['network']['input_shape'].split(',')])
        self.choose_net = config['network']['net']
        self.eta = config['network'].getfloat('learning_rate')  # Learning rate
        self.gamma = config['network'].getfloat('discount_rate')  # Discount rate
        self.decay = config['network'].getfloat('learning_decay')
        self.epsilon = config['network'].getfloat('epsilon')
        self.epsilon_min = config['network'].getfloat('epsilon_min')
        self.epsilon_decay = config['network'].getfloat('epsilon_decay')
        self.batch_size = config['network'].getint('batch_size')
        self.pretraining = config['simulation'].getboolean('pretraining')
        self.epoch = config['network'].getint('epoch')
        self.threshold = config['network'].getfloat('threshold')
        self.one_hot = config['network'].getboolean('one_hot_encoding')
        self.evaluate = 0
        self.number_of_cubes_to_solve = 100

        # Weights
        self.load_weights = config['network'].getboolean('load_weights')
        self.load_model_path = config['network']['load_model']

        # Simulation
        self.done = 1                               # Checks if a cube is done
        self.solved = 0                             # Calculates the current accuracy
        self.epsilon_decay_steps = 0                # Decreases exploration after more steps is incremented
        self.simulations_this_scrambles = 0         # How many simulations has been done for the current difficulty

        self.memory = deque(maxlen=self.batch_size) # Length of memory

        self.difficulty_level = 1                   # The amount of scrambles
        self.best_accuracy = 0.0                    # The best accuracy for the current difficulty
        self.difficulty_counter = 0                 # Current amount of scrambles
        self.accuracy_points = []                   # Y-axis for plot
        self.axis = []                              # X-axis for plot
        self.plot_progress = config['simulation'].getboolean('show_plot')  # Plots the progress
        self.check_level = 1
        self.difficulty_test = config['simulation'].getint('difficulty_level_test')
        self.number_of_moves_eval = []
        self.render_image = config['environment'].getboolean('render_image')

        # Initialize network
        # Load a model if in test mode or user wants to train from an existing net
        # If not, then a new net is initiated for training.
        if self.load_weights is True or self.test_model is True:
            self.network = self.load_network()
        else:
            if self.choose_net == 'fcn':
                self.network = self.model_fcn()
            elif self.choose_net == 'conv':
                self.network = self.model_conv()

        # If one hot, change the input shape to match the extra added dimension. Does not work with conv net.
        if self.one_hot:
            if self.choose_net == 'fcn':
                self.input_shape = 1, 144
            elif self.choose_net =='conv':
                self.input_shape = 1, 6, 2, 12


    def model_fcn(self):
        """
        Creates a Fully Connected neural network
        Input shape is 1x144
        Returns the expected reward for a set of actions.
        :return:
        """

        model = keras.models.Sequential()

        model.add(keras.layers.Dense(4096, activation='relu',
                                     batch_size=self.batch_size,
                                     ))

        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(4096, activation='relu'
                                     ))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(2096, activation='relu'
                                     ))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(512, activation='relu'
                                     ))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(12, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,

                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model

    def model_conv(self):
        """
        Creates a convolutional neural network.
        Input shape is 6x2x2x12 array and returns.
        Returns the expected reward for a set of actions.
        :return:
        """

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(256, kernel_size=(11, 11), strides=(4, 4), activation=tf.nn.relu,
                                          batch_size=self.batch_size, padding='same'
                                          ))
        model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(3, 3), activation=tf.nn.relu,
                                      padding='same'
                                      ))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(1024, activation='relu',

                                     ))
        model.add(keras.layers.Dense(512, activation='relu',
                                     ))

        model.add(keras.layers.Dense(12, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
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
                         batch_size=self.batch_size
                         )

    def load_network(self):
        """
        Loads a pretrained network
        :return:
        """
        return keras.models.load_model(self.load_model_path)

    def train(self, cube):
        """
        Trains the data with reinforcement learning.
        Loops through episodes and stores experience to memory
        :param cube:
        :return:
        """

        solved_rate = deque(maxlen=self.number_of_cubes_to_solve)

        # Check the last difficulty level if loading weights
        if self.load_weights is True:
            # Find the difficulty level
            self.difficulty_level = int(re.search('_(\d)_', self.load_model_path)[0].strip('_'))

        simulation = 0

        # Start episodes
        while True:
                try:
                    memory_temp = deque(maxlen=self.difficulty_level)
                    # Reset cube before each simulation
                    cube.cube, cube.face = cube.reset_cube()

                    # Scramble the cube as many times as the scramble_limit
                    _, scramble_actions = cube.scramble_cube(self.difficulty_level)

                    if cube.reward() == self.done:
                        continue

                    # After 1000 simulations, pre-training is turned off
                    if simulation >= 1000:
                        self.pretraining = False

                    for step in range(self.difficulty_level):

                        # Get the state of the cube
                        state = copy.deepcopy(cube.cube)

                        if self.one_hot:
                            # One hot encoding
                            state = keras.utils.to_categorical(state)

                        # Reshaping the state
                        state = state.reshape(self.input_shape)

                        # Take in state and predict the reward for all possible actions
                        actions = self.network.predict(state)

                        # Either choose predicted action or explore depending on current epsilon value
                        if self.pretraining is True or self.get_epsilon(self.epsilon_decay_steps) >= np.random.random():
                            self.epsilon_decay_steps += 1
                            take_action = np.random.random_integers(0, 11, 1)
                        else:
                            take_action = np.argmax(actions)

                        # Execute action v
                        next_state, face = cube.rotate_cube(take_action, render_image=False)

                        # Calculate reward and find the next state
                        reward = cube.reward()

                        if self.one_hot:
                            next_state = keras.utils.to_categorical(next_state)

                        # Append the result into the dataset
                        memory_temp.appendleft((state.reshape(self.input_shape), actions, reward, next_state))

                        # Is the cube solved?
                        if reward == self.done:
                            solved_rate.appendleft(1)
                            break

                    # Go train, if pretraining is finished
                    for train in range(self.epoch):
                        if len(self.memory) >= self.batch_size:
                            self.go_to_gym()

                    # Increase the simulation counter
                    simulation += 1
                    self.simulations_this_scrambles += 1
                    if simulation % 5000 == 0:
                        keras.models.save_model(self.network,
                                                f"models/training_{self.difficulty_level}_scrambles.h5")

                    # If the reward is zero here, it means the cube was not solved
                    if reward != self.done:
                        solved_rate.appendleft(0)

                    # Add episode to memory and create target vectors
                    self.add_to_memory(memory_temp)

                    # Print out the current progress
                    self.check_progress(simulation, solved_rate)

                    # Evaluate the network
                    self.evaluate_network(cube)

                # Save the current model when exiting
                except KeyboardInterrupt:
                    keras.models.save_model(self.network,
                                           f"models/solves_{self.difficulty_level}_scrambles - {time.time()}.h5")
                    break

    def add_to_memory(self, memory_temp):
        """
        Adds memory_tempt to memory and corrects the target vector.
        :param memory_temp:
        :return:
        """
        # Get the end reward to set up the target vector
        end_reward = memory_temp[0][2]

        for state, actions, reward, next_state in memory_temp:
            take_action = np.argmax(actions)

            # Create the target from the reward and predicted reward
            target_vector = self.create_target_vector(next_state, reward, actions, take_action, end_reward)

            self.memory.appendleft((state, actions, target_vector, next_state))

    def create_target_vector(self, next_state, reward, actions, take_action, end_reward):
        """
        Creates the target vector used as label in the training.
        Will also look at future rewards with a discount factor
        :param end_reward:
        :param take_action:
        :param next_state:
        :param reward:
        :param actions:
        :return:
        """
        # If we know the final reward is 1, then the action yields a rewards as well
        if end_reward == 1:
            check_future = self.network.predict(next_state.reshape(self.input_shape))
            target = reward + self.gamma*np.argmax(check_future)
            # target = 1

        # If there is no reward in the end, then the current reward is 0
        else:
            target = reward

        actions[0][take_action] = target
        return actions

    def sample_memory(self):
        """
        Creates a mini batch from the memory. Samples uniformly
        :return:
        """
        # Sample a batch of data from memory
        mini_batch = random.sample(self.memory, self.batch_size)
        # mini_batch = self.memory
        # Append the state and target from the data
        x_batch, y_batch = [], []
        for state, act, target, next_state in mini_batch:
            x_batch.append(state), y_batch.append(target)

        # Reshape the data to match the batch size
        lst = list(self.input_shape)
        lst[0] = self.batch_size
        lst = tuple(lst)
        states = np.array(x_batch).reshape(lst)
        rewards = np.array(y_batch).reshape(self.batch_size, 12)
        return states, rewards

    def get_epsilon(self, decay_steps):
        """
        Returns the e - greedy based on how many simulations that has been run for this
        amount of scrambles
        :param decay_steps: The number of simulations used for the current amount of scrambles.
        :return:
        """
        if self.pretraining is False:
            return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((decay_steps + 1) * self.epsilon_decay)))
        else:
            return 1

    def plot_accuracy(self, simulation, solved, solved_rate):
        """
        Plots the accuracy of the cubes
        :return:
        """
        # Plot the accuracy
        if simulation % len(solved_rate) * 10 == 0:
            if self.difficulty_level > self.check_level:
                if len(self.axis) > 0:
                    plt.savefig(f"Accuracy_{self.difficulty_level-1}.png"), plt.close()
                self.check_level = self.difficulty_level
                self.axis.clear(), self.accuracy_points.clear()

            self.axis.append(simulation)
            self.accuracy_points.append(solved)
            plt.plot(self.axis, self.accuracy_points, color='r')
            plt.ylim((0, 1.3))
            plt.xlabel('Simulations')
            plt.ylabel('Solved cubes')
            plt.title(f'Accuracy plot for {self.difficulty_level} scramble')
            plt.pause(0.001)

    def check_progress(self, simulation, solved_rate):
        """
        Checks the progress of the training and increases the number of scrambles if it deems
        the network good enough
        :param simulation:
        :param solved_rate:
        :return:
        """
        # Calculate score
        if simulation % len(solved_rate) == 0:
            self.solved = (sum(solved_rate) / len(solved_rate))

            # Print the current progress
            print(f"\033[93m"     
                  f"{sum(solved_rate)} Cubes solved of the last {len(solved_rate)}"
                  f" \n\033[35mAccuracy: {round(self.solved, 2)}\033[0m"
                  f"\033[93m\nExploration rate: {round(self.get_epsilon(self.epsilon_decay_steps), 5)}"
                  f"\nScrambles: {self.difficulty_level} \n\033[35mNumber of simulations: {simulation}\033[0m"
                  f"\n\033[93mBest accuracy: {round(self.best_accuracy, 3)}, "
                  f"reached 100% {self.difficulty_counter} times\033[0m"
                  f"\n\033[93m=================================\033[0m"
                  f"\033[0m")

            # Update the best accuracy
            if self.solved > self.best_accuracy:
                self.best_accuracy = self.solved

            if self.plot_progress is True:
                self.plot_accuracy(simulation, self.solved, solved_rate)

    def evaluate_network(self, cube):
        """
        Evaluates how well the network is performing over all.
        :return:
        """
        # Saves the model if the model is deemed good enough
        if round(self.solved, 2) >= 0.99 and self.pretraining is False:
            self.difficulty_counter += 1
            self.number_of_moves_eval = []
            rewards = self.test(cube)


            accuracy = sum(rewards)/len(rewards)
            print('\n')
            if accuracy >= self.threshold:

                # Increment difficulty level
                self.epsilon = config['network'].getfloat('epsilon')

                print(f"\033[91m"
                      f"Solved {sum(rewards)}/{len(rewards)} with an accuracy of {accuracy}"
                      f"\nAverage number of moves used: {np.mean(self.number_of_moves_eval)}"
                      f"\nIncreasing the number of scrambles by 1"
                      f"\033[0m"
                      f"\n\033[93m=================================\033[0m")

                # Reset variables
                self.best_accuracy = 0
                self.simulations_this_scrambles = 0
                self.epsilon_decay_steps = 0
                self.solved = 0
                self.difficulty_counter = 0


                self.difficulty_level += 1
                if config['simulation'].getboolean('test') is not True:
                    keras.models.save_model(self.network,
                                            f"models/solves_{self.difficulty_level}_scrambles - {time.time()}.h5")

            # Return to training if the threshold has not been met
            else:
                self.solved = 0
                print(f'\033[92m'
                      f'Total accuracy is {accuracy}, \nneeded {self.threshold}. Return to training'
                      f"\nAverage number of moves used: {np.mean(self.number_of_moves_eval)}"
                      f'\033[0m')

    def test(self, cube, simulations=1000):
        """
        Method for testing a trained model
        :param cube: The environment
        :param simulations: Number of simulations
        :return:
        """

        # Initiate variables
        rewards = []
        scrambles = []
        self.pretraining = False

        print("\033[93mEvaluating:\033[0m")
        # Start looping through episodes
        for simulation in range(simulations):

            # Reset cube before each simulation
            cube.cube, cube.face = cube.reset_cube()

            # Scramble the cube as many times as the scramble_limit

            _, scramble_actions = cube.scramble_cube(self.difficulty_level, render_image=self.render_image)

            if cube.reward() == self.done:
                continue

            # Loop through episode

            for step in range(self.difficulty_level):

                # Get the state of the cube
                state = copy.deepcopy(cube.cube)

                if self.one_hot:
                    # One hot encoding
                    state = keras.utils.to_categorical(state)

                # Reshaping the state
                state = state.reshape(self.input_shape)

                # Predict best action
                actions = self.network.predict(state)

                # Choose predicted action or explore.
                take_action = np.argmax(actions)

                # Execute action
                cube.rotate_cube(take_action, render_image=False)

                # Calculate reward and find the next state
                reward = cube.reward()

                # Is the cube solved?
                if reward == 1:
                    self.number_of_moves_eval.append(step + 1)
                    rewards.append(1)
                    scrambles.append(step)
                    break

            # Increase the simulation counter
            simulation += 1

            # If the reward is zero here, it means the cube was not solved
            if reward == 0:
                rewards.append(0)

            sys.stdout.write('\r\033[35m' + str(simulation)+f'/{simulations} cubes\033[0m')
            sys.stdout.flush()

        return rewards


def run_network_test(model, rubiks_cube):
    """
    Runs an evaluation of a previously trained model
    :param model:
    :param rubiks_cube:
    :return:
    """
    x_axis = []
    y_axis = []
    # model.difficulty_level = 14
    for i in range(model.difficulty_test):
        rewards = model.test(rubiks_cube, 1000)
        accuracy = sum(rewards) / len(rewards)
        print('\n')
        print(f"\033[94m"
              f"Solved {sum(rewards)}/{len(rewards)} with an accuracy of {accuracy}"
              f"\nDifficulty was {model.difficulty_level} scrambles"
              f"\033[0m"
              f"\n\033[93m=================================\033[0m")
        x_axis.append(i + 1), y_axis.append(accuracy * 100)
        model.difficulty_level += 1
    plt.plot(x_axis, y_axis, color='r', marker='o')
    plt.ylim((0, 110)), plt.xlim(0, model.difficulty_test + 1)
    plt.xlabel('Number of scrambles'), plt.ylabel('Solve percentage')
    plt.title('Solve percentage at different scramble lengths')
    plt.show()


def main():
    """
    Main function of the script
    :return:
    """
    # Set up environment
    rubiks_cube = Cube()

    # Initialize network
    model = Network(rubiks_cube.cube)

    # Start training
    if model.train_model is True:
        model.train(rubiks_cube)

    # Start testing
    elif model.test_model is True:
        run_network_test(model, rubiks_cube)


if __name__ == '__main__':
    main()

















