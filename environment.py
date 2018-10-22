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

    def __init__(self, cube_size=(6, 2, 2)):
        """

        :param cube_size: Cube size is 2x2 by default
        """

        self.cube_size = cube_size

        self.cube = np.empty(self.cube_size, dtype=str)
        colors = ['w', 'b', 'o', 'r', 'g', 'y']
        for k, color in enumerate(colors):
            self.cube[k] = color

        self.face = {'U': self.cube[0], 'D': self.cube[1],
                     'F': self.cube[4], 'B': self.cube[5],
                     'R': self.cube[3], 'L': self.cube[2]}

    def move(self, face, dir):
        """

        Move the faces of the cube in either clock-wise or counter-clock-wise
        :param face: The face to rotate
        :param dir: The direction. -1 is clockwise, 1 is counter clockwise
        :return:
        """
        f = self.face[face]

        if f == 0:
            np.rot90(self.cube[0], dir)



    def __repr__(self) -> str:
        return str(self.cube)



if __name__ == "__main__":

    environment = Cube()
    print(environment)
    environment.move(0, 1)
    # cube = np.empty((6, 2, 2), dtype=str)
    # liste = ['w', 'o', 'r', 'g', 'b', 'y']
    # for i, farge in enumerate(liste):
    #     cube[i] = farge
    #
    # print(cube)


    # rotate = np.array(([1, 2, 3], [8, 0, 4], [7, 6, 5]))
    # print(rotate)
    # print(np.transpose(rotate))
    # print(np.rot90(rotate, -1))



