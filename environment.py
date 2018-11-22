"""
This is the environment script for the agent.
"""

import numpy as np
from color_cube import make_plot
import configparser
import random

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
        self.action_space = np.array([['L', 'Lr'], ['U', 'Ur'], ['F', 'Fr'],
                                      ['D', 'Dr'], ['R', 'Rr'], ['B', 'Br']])

        self.num_to_str = {0: 'L', 6: 'Lr', 1: 'U', 7: 'Ur', 2: 'F', 8: 'Fr',
                           3: 'D', 9: 'Dr', 4: 'R', 10: 'Rr', 5: 'B', 11: 'Br'}


    def rotate_cube(self, face, render_image=False):
        """
        Move the faces of the cube in either clock-wise or counter-clock-wise
        Movement is hardcoded to work on 2x2x6 cubes
        :param face: The face to rotate
        :param dir: The direction. -1 is clockwise, 1 is counter clockwise
        :param render_image: If True, render an image before and after the move
        :return:
        """
        numeric = config['environment'].getboolean('numeric_representation')
        cube_temp = self.cube.copy()
        # L = 0, U = 1, F = 2, D = 3, R = 4, B = 5

        if face != str(face):
            face = self.num_to_str[int(face)]

        if render_image:
            make_plot(self.cube, numeric)

        if face == 'F':
            self.face['R'][0, 0], self.face['R'][1, 0] = cube_temp[1][1, 0], cube_temp[1][1, 1]
            self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[4][1, 0], cube_temp[4][0, 0]
            self.face['L'][0, 1], self.face['L'][1, 1] = cube_temp[3][0, 0], cube_temp[3][0, 1]
            self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[0][1, 1], cube_temp[0][0, 1]

            self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[2][1, 0], cube_temp[2][0, 0]
            self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[2][1, 1], cube_temp[2][0, 1]
        elif face == 'Fr':
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
        elif face == 'Br':
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
        elif face == 'Rr':
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
        elif face == 'Lr':
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
        elif face == 'Ur':
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
        elif face == 'Dr':
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
            make_plot(self.cube, numeric)

        return self.cube, face

    def reset_cube(self):
        """
        Sets a cube array to start position
        :return:
        """

        if config['environment'].getboolean('numeric_representation'):
            cube = np.zeros(self.cube_size, dtype=int)
            colors = [1, 5, 3, 0, 2, 4]
            if config['environment'].getboolean('random_sides') is True:
                random.shuffle(colors)
        else:
            cube = np.zeros(self.cube_size, dtype=str)
            colors = ['g', 'y', 'o', 'w', 'b', 'r']
        for k, color in enumerate(colors):
            cube[k] = color
        face = {'L': cube[0], 'U': cube[1],
                'F': cube[2], 'D': cube[3],
                'R': cube[4], 'B': cube[5]}
        return cube, face

    def scramble_cube(self, k: int, render_image=False):
        """
        Takes in a cube array, and scramble it k times.
        Returns the scrambled cube
        :param k:
        :return:
        """
        store_action = []  # List to store the actions

        # Randomly choose actions from the action space

        action = np.random.choice(range(0, 12), size=k)

        # Loop through actions and move the cube
        for a in action:
            if config['environment'].getboolean('expand_training') is True:
                if store_action != []:
                    while a == store_action[-1]+6 or a == store_action[-1]-6:
                        a = np.random.choice(range(0, 12), size=1)

            store_action.append(a)
            self.rotate_cube(a, render_image)

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
    cube.rotate_cube('F', True)