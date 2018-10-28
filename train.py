import keras
import tensorflow

# TODO Consider making a main class for models, with subclasses for reinforcement and supervised

# TODO Since we have 3.600.000 million combinations, perhaps it is possible to completely use supervise learning
# TODO A possible reward could be how many moves we are away from the solution.


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

def main():


    pass
















