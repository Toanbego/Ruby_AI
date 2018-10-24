"""
This is the agent.
"""
import environment


class Solver:

    def __init__(self, state):

        # self.end_state = environment.cube
        self.state = state
        self.reward = self.reward()

    def action(self, face, dir):
        cube.rotate_cube(face=face, dir=dir)

    def reward(self, state, action, next_state='noe'):
        """
        Should calculate reward based on some of this information.
        Perhaps the number of moves it is away from wind state
        :param state:
        :param action:
        :param next_state:
        :return:
        """
        if state == self.end_state:
            return 1
        else:
            return -1



if __name__ == "__main__":

    cube = environment.Cube()

    agent = Solver(cube)
    agent.action('F', 1)
    print(cube)
    agent.action('F', -1)

    # Calculate reward
    reward = agent.reward(cube, 'F')

    print(cube)



