"""
This is the environment script for the agent.
"""

import numpy as np
from color_cube import make_image
import configparser
config = configparser.ConfigParser()
config.read("config.ini")


class Cube:
    """
    Class for representing the Rubik's Cube

    Attributes:
        cube_size: The integer will represent a quadratic number. If cube_size = 2, then the cube is a 6x2x2
        cube: 6x2x2 array containing all the faces
        face: Dict with all the faces stored as arrays
    """

    def __init__(self, cube_size=(6, 2, 2)):
        """
        :param cube_size: Cube size is 2x2 by default
        """

        self.cube_size = cube_size

        self.cube, self.face = self.reset_cube()
        self.action_space = ["L", "L'", "U", "U'", "F", "F'",
                             "D", "D'", "R", "R'", "B", "B'"]

    def rotate_cube(self, face, render_image=True):
        """
        Move the faces of the cube in either clock-wise or counter-clock-wise
        Movement is hardcoded to work on 2x2x6 cubes
        :param face: The face to rotate
        :param dir: The direction. -1 is clockwise, 1 is counter clockwise
        :param render_image: If True, render an image before and after the move
        :return:
        """

        cube_temp = self.cube.copy()
        # L = 0, U = 1, F = 2, D = 3, R = 4, B = 5
        if render_image:
            make_image(self.cube, 'before_move')
        if face == 'F':
            self.face['R'][0, 0], self.face['R'][1, 0] = cube_temp[1][1, 0], cube_temp[1][1, 1]
            self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[4][1, 0], cube_temp[4][0, 0]
            self.face['L'][0, 1], self.face['L'][1, 1] = cube_temp[3][0, 0], cube_temp[3][0, 1]
            self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[0][1, 1], cube_temp[0][0, 1]

            self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[2][1, 0], cube_temp[2][0, 0]
            self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[2][1, 1], cube_temp[2][0, 1]
        elif face == "F'":
            self.face['R'][0, 0], self.face['R'][1, 0] = cube_temp[3][0, 1], cube_temp[3][0, 0]
            self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[0][0, 1], cube_temp[0][1, 1]
            self.face['L'][0, 1], self.face['L'][1, 1] = cube_temp[1][1, 1], cube_temp[1][1, 0]
            self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[4][0, 0], cube_temp[4][1, 0]

            self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[2][0, 1], cube_temp[2][1, 1]
            self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[2][0, 0], cube_temp[2][1, 0]

        elif face == 'B':
            self.face['L'][0, 0], self.face['L'][1, 0] = cube_temp[1][0, 1], cube_temp[1][0, 0]
            self.face['D'][1, 0], self.face['D'][1, 1] = cube_temp[0][0, 0], cube_temp[0][1, 0]
            self.face['R'][0, 1], self.face['R'][1, 1] = cube_temp[3][1, 1], cube_temp[3][1, 0]
            self.face['U'][0, 0], self.face['U'][0, 1] = cube_temp[4][0, 1], cube_temp[4][1, 1]

            self.face['B'][0, 0], self.face['B'][0, 1] = cube_temp[5][1, 0], cube_temp[5][0, 0]
            self.face['B'][1, 0], self.face['B'][1, 1] = cube_temp[5][1, 1], cube_temp[5][0, 1]
        elif face == "B'":
            self.face['L'][0, 0], self.face['L'][1, 0] = cube_temp[3][1, 0], cube_temp[3][1, 1]
            self.face['D'][1, 0], self.face['D'][1, 1] = cube_temp[4][1, 1], cube_temp[4][0, 1]
            self.face['R'][0, 1], self.face['R'][1, 1] = cube_temp[1][0, 0], cube_temp[1][0, 1]
            self.face['U'][0, 0], self.face['U'][0, 1] = cube_temp[0][1, 0], cube_temp[0][0, 0]

            self.face['B'][0, 0], self.face['B'][0, 1] = cube_temp[5][0, 1], cube_temp[5][1, 1]
            self.face['B'][1, 0], self.face['B'][1, 1] = cube_temp[5][0, 0], cube_temp[5][1, 0]

        elif face == 'R':
            self.face['B'][0, 0], self.face['B'][1, 0] = cube_temp[1][1, 1], cube_temp[1][0, 1]
            self.face['D'][0, 1], self.face['D'][1, 1] = cube_temp[5][1, 0], cube_temp[5][0, 0]
            self.face['F'][0, 1], self.face['F'][1, 1] = cube_temp[3][0, 1], cube_temp[3][1, 1]
            self.face['U'][0, 1], self.face['U'][1, 1] = cube_temp[2][0, 1], cube_temp[2][1, 1]

            self.face['R'][0, 0], self.face['R'][0, 1] = cube_temp[4][1, 0], cube_temp[4][0, 0]
            self.face['R'][1, 0], self.face['R'][1, 1] = cube_temp[4][1, 1], cube_temp[4][0, 1]
        elif face == "R'":
            self.face['B'][0, 0], self.face['B'][1, 0] = cube_temp[3][1, 1], cube_temp[3][0, 1]
            self.face['D'][0, 1], self.face['D'][1, 1] = cube_temp[2][0, 1], cube_temp[2][1, 1]
            self.face['F'][0, 1], self.face['F'][1, 1] = cube_temp[1][0, 1], cube_temp[1][1, 1]
            self.face['U'][0, 1], self.face['U'][1, 1] = cube_temp[5][1, 0], cube_temp[5][0, 0]

            self.face['R'][0, 0], self.face['R'][0, 1] = cube_temp[4][0, 1], cube_temp[4][1, 1]
            self.face['R'][1, 0], self.face['R'][1, 1] = cube_temp[4][0, 0], cube_temp[4][1, 0]

        elif face == 'L':
            self.face['F'][0, 0], self.face['F'][1, 0] = cube_temp[1][0, 0], cube_temp[1][1, 0]
            self.face['D'][0, 0], self.face['D'][1, 0] = cube_temp[2][0, 0], cube_temp[2][1, 0]
            self.face['B'][0, 1], self.face['B'][1, 1] = cube_temp[3][1, 0], cube_temp[3][0, 0]
            self.face['U'][0, 0], self.face['U'][1, 0] = cube_temp[5][1, 1], cube_temp[5][0, 1]

            self.face['L'][0, 0], self.face['L'][0, 1] = cube_temp[0][1, 0], cube_temp[0][0, 0]
            self.face['L'][1, 0], self.face['L'][1, 1] = cube_temp[0][1, 1], cube_temp[0][0, 1]
        elif face == "L'":
            self.face['F'][0, 0], self.face['F'][1, 0] = cube_temp[3][0, 0], cube_temp[3][1, 0]
            self.face['D'][0, 0], self.face['D'][1, 0] = cube_temp[5][1, 1], cube_temp[5][0, 1]
            self.face['B'][0, 1], self.face['B'][1, 1] = cube_temp[1][1, 0], cube_temp[1][0, 0]
            self.face['U'][0, 0], self.face['U'][1, 0] = cube_temp[2][0, 0], cube_temp[2][1, 0]

            self.face['L'][0, 0], self.face['L'][0, 1] = cube_temp[0][0, 1], cube_temp[0][1, 1]
            self.face['L'][1, 0], self.face['L'][1, 1] = cube_temp[0][0, 0], cube_temp[0][1, 0]

        elif face == 'U':
            self.face['R'][0, 0], self.face['R'][0, 1] = cube_temp[5][0, 0], cube_temp[5][0, 1]
            self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[4][0, 0], cube_temp[4][0, 1]
            self.face['L'][0, 0], self.face['L'][0, 1] = cube_temp[2][0, 0], cube_temp[2][0, 1]
            self.face['B'][0, 0], self.face['B'][0, 1] = cube_temp[0][0, 0], cube_temp[0][0, 1]

            self.face['U'][0, 0], self.face['U'][0, 1] = cube_temp[1][1, 0], cube_temp[1][0, 0]
            self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[1][1, 1], cube_temp[1][0, 1]
        elif face == "U'":
            self.face['R'][0, 0], self.face['R'][0, 1] = cube_temp[2][0, 0], cube_temp[2][0, 1]
            self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[0][0, 0], cube_temp[0][0, 1]
            self.face['L'][0, 0], self.face['L'][0, 1] = cube_temp[5][0, 0], cube_temp[5][0, 1]
            self.face['B'][0, 0], self.face['B'][0, 1] = cube_temp[4][0, 0], cube_temp[4][0, 1]

            self.face['U'][0, 0], self.face['U'][0, 1] = cube_temp[1][0, 1], cube_temp[1][1, 1]
            self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[1][0, 0], cube_temp[1][1, 0]

        elif face == 'D':
            self.face['R'][1, 0], self.face['R'][1, 1] = cube_temp[2][1, 0], cube_temp[2][1, 1]
            self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[0][1, 0], cube_temp[0][1, 1]
            self.face['L'][1, 0], self.face['L'][1, 1] = cube_temp[5][1, 0], cube_temp[5][1, 1]
            self.face['B'][1, 0], self.face['B'][1, 1] = cube_temp[4][1, 0], cube_temp[4][1, 1]

            self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[3][1, 0], cube_temp[3][0, 0]
            self.face['D'][1, 0], self.face['D'][1, 1] = cube_temp[3][1, 1], cube_temp[3][0, 1]
        elif face == "D'":
            self.face['R'][1, 0], self.face['R'][1, 1] = cube_temp[5][1, 0], cube_temp[5][1, 1]
            self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[4][1, 0], cube_temp[4][1, 1]
            self.face['L'][1, 0], self.face['L'][1, 1] = cube_temp[2][1, 0], cube_temp[2][1, 1]
            self.face['B'][1, 0], self.face['B'][1, 1] = cube_temp[0][1, 0], cube_temp[0][1, 1]

            self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[3][0, 1], cube_temp[3][1, 1]
            self.face['D'][1, 0], self.face['D'][1, 1] = cube_temp[3][0, 0], cube_temp[3][1, 0]

        # Update cube array
        for i, face in enumerate(self.face):
            self.cube[i] = self.face[face]

        if render_image:
            make_image(self.cube, 'after_move')

        return self.cube

    def reset_cube(self):
        """
        Sets a cube array to start position
        :return:
        """
        cube = np.zeros(self.cube_size, dtype=str)
        colors = ['g', 'y', 'o', 'w', 'b', 'r']
        for k, color in enumerate(colors):
            cube[k] = color
        face = {'L': cube[0], 'U': cube[1],
                'F': cube[2], 'D': cube[3],
                'R': cube[4], 'B': cube[5]}
        return cube, face

    def scramble_cube(self, k: int):
        """
        Takes in a cube array, and scramble it k times.
        Returns the scrambled cube
        :param k:
        :return:
        """
        store_action = []  # List to store the actions

        # Randomly choose actions from the action space
        action = np.random.choice(self.action_space, size=k)

        # Loop through actions and move the cube

        for a in action:
            store_action.append(a)
            self.rotate_cube(a)

        return [self.cube, store_action]

    def __repr__(self) -> str:
        """
        Prints a string representing the array cube
        :return:
        """
        rep_string = ''
        for faces in self.face:
            rep_string += f'\n===={faces}====\n{self.face[faces]}'
        return rep_string


if __name__ == "__main__":
    cube = Cube()
    data_set = []
    # cube.scramble_cube(10)
    # cube.cube, cube.face = cube.reset_cube()
    #
    # cube.rotate_cube('U', True)
    # cube.rotate_cube("F'", True)
    # cube.rotate_cube('R', -1)
    # cube.rotate_cube('B', -1)
    # cube.rotate_cube('D', -1)
    # cube.rotate_cube('L', -1)

    # cube.rotate_cube('L', 1)
    # cube.rotate_cube('B', 1)
    # cube.rotate_cube('R', 1)
    # cube.rotate_cube('D', 1)
    # cube.rotate_cube('F', 1)
    # cube.rotate_cube('U', 1)
