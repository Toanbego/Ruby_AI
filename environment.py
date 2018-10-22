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
        colors = ['w', 'b', 'r', 'g', 'o', 'y']
        for k, color in enumerate(liste):
            self.cube[k] = color

        self.up = np.zeros(cube_size, dtype=str)
        self.up[:] = 'w'
        self.down = np.zeros(cube_size, dtype=str)
        self.down[:] = 'b'
        self.left = np.zeros(cube_size, dtype=str)
        self.left[:] = 'o'
        self.right = np.zeros(cube_size, dtype=str)
        self.right[:] = 'r'
        self.front = np.zeros(cube_size, dtype=str)
        self.front[:] = 'g'
        self.back = np.zeros(cube_size, dtype=str)
        self.back[:] = 'y'
        self.face = {'U': cube[0], 'D': self.down,
                     'F': self.front, 'B': self.back,
                     'R': self.right, 'L': self.left}

    def move(self, face, dir):
        """

        Move the faces of the cube in either clock-wise or counter-clock-wise
        :param face: The face to rotate
        :param dir: The direction. -1 is clockwise, 1 is counter clockwise
        :return:
        """
        f = self.face[face]

        if f == 'F':
            np.rot90(self.face[face], dir)



    def __repr__(self) -> str:
        return str(self.face)



if __name__ == "__main__":

    # environment = Cube()
    # print(environment)
    # environment.move('F', 'c')
    cube = np.empty((6, 2, 2), dtype=str)
    liste = ['w', 'o', 'r', 'g', 'b', 'y']
    for i, farge in enumerate(liste):
        cube[i] = farge

    print(cube)


    # rotate = np.array(([1, 2, 3], [8, 0, 4], [7, 6, 5]))
    # print(rotate)
    # print(np.transpose(rotate))
    # print(np.rot90(rotate, -1))



