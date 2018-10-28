"""
This is the agent.
"""
import environment


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

        self.end_state = state
        # self.reward = self.reward(self.end_state)

    def action(self, face, dir):
        """
        Perform an action on the rubik's cube environment according to a given
        policy
        :param face:
        :param dir:
        :return:
        """
        rubiks_cube.rotate_cube(face=face, dir=dir)


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


def generate_training_samples(data_set_size, scrambles, rubiks_cube):
    """
    Generate the training set for supervised learning
    :param data_set_size: size of data set
    :param scrambles: How many times to maximum scramble a cube
    :param rubiks_cube: Cube object
    :return:
    """
    # TODO This should probably be done beforehand and written to a datafile
    # TODO It might also be an idea to not make the dataset randomly scrambled, but deterministic
    data_set = []   # Will consist of the cube array

    # Generate randomly scrambled cubes
    for sample in range(data_set_size):

        # Reset the cube each time
        rubiks_cube.cube, rubiks_cube.face = rubiks_cube.reset_cube()

        # Append the scrambled cube, with the actions it took to get there
        data_set.append(rubiks_cube.scramble_cube(10))

    return data_set


if __name__ == "__main__":

    # Set up environment
    rubiks_cube = environment.Cube()

    # Generate Training set
    data = generate_training_samples(100, 1, rubiks_cube)

    # Initialize agent
    agent = Solver(rubiks_cube.cube)

    # Testing some methods

    # agent.action('F', 1)
    # agent.action('F', -1)
    # # Calculate reward
    # reward = agent.reward(rubiks_cube.cube)
    # print(reward)





