"""
This is the environment script for the agent.
"""

import numpy as np


class Cube:
    """
    Class for representing the Rubik's Cube

    Attributes:
        cube_size: THe integer will represent a quadratic number. If cube_size = 2, then the cube is a 6x2x2

    """

    def __init__(self, cube_size=2):
        """

        :param cube_size: Cube size is 2x2 by default
        """
        self.face = {'U': 0, 'D': 1, 'F': 2, 'B': 3, 'R': 4, 'L': 5}
        self.cube_size = np.zeros((6, cube_size, cube_size))

    def __repr__(self) -> str:

        return str(self.cube_size), self.face



if __name__ == "__main__":
    environment = Cube()
    print(environment)

