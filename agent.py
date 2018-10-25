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


if __name__ == "__main__":

    rubiks_cube = environment.Cube()

    # Testing some methods
    agent = Solver(rubiks_cube.cube)
    agent.action('F', 1)
    agent.action('F', -1)
    # Calculate reward
    reward = agent.reward(rubiks_cube.cube)
    print(reward)





