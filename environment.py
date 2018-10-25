"""
This is the environment script for the agent.
"""

import numpy as np
import copy


class Cube:
    """
    Class for representing the Rubik's Cube

    Attributes:
        cube_size: The integer will represent a quadratic number. If cube_size = 2, then the cube is a 6x2x2

    """
    """
    F = front[0,1,2,3] = front[2,0,3,1], right[0,2] = up[2,3], down[0,1] = right[0,2]
        left[1,3] = down[0,1]
        
    
    """

    def __init__(self, cube_size=(6, 2, 2)):
        """
        :param cube_size: Cube size is 2x2 by default
        """

        self.cube_size = cube_size
        self.cube = np.zeros(self.cube_size, dtype=str)

        # This is the classical colors
        # colors = ['w', 'b', 'o', 'r', 'g', 'y']
        # for k, color in enumerate(colors):
        #     self.cube[k] = color

        colors = ['w', 'y', 'b', 'g', 'r', 'o']
        for k, color in enumerate(colors):
            self.cube[k] = color

        # This is for numbering the array instead of with str
        # self.cube = np.array([[[0,1],[2,3]], [[0,1],[2,3]], [[0,1],[2,3]], [[0,1],[2,3]], [[0,1],[2,3]], [[0,1],[2,3]]])

        # Create dictionary from array.
        self.face = {'U': self.cube[0], 'D': self.cube[1],
                     'L': self.cube[2], 'R': self.cube[3],
                     'F': self.cube[4], 'B': self.cube[5]}

    def rotate_cube(self, face, dir):
        """
        Move the faces of the cube in either clock-wise or counter-clock-wise
        Movement is hardcoded to work on 2x2x6 cubes
        :param face: The face to rotate
        :param dir: The direction. -1 is clockwise, 1 is counter clockwise
        :return:
        """

        cube_temp = self.cube.copy()

        # U=0, D=1, L=2, R=3, F=4, B=5

        if face == 'F':
            self.face['F'] = np.rot90(self.face['F'], dir)
            if dir == -1:
                self.face['R'][0, 0], self.face['R'][1, 0] = cube_temp[0][1, 0], cube_temp[0][0, 1]
                self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[3][0, 0], cube_temp[3][1, 0]
                self.face['L'][0, 1], self.face['L'][1, 1] = cube_temp[1][0, 0], cube_temp[1][0, 1]
                self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[2][0, 1], cube_temp[2][1, 1]
            else:
                self.face['R'][0, 0], self.face['R'][1, 0] = cube_temp[1][0, 0], cube_temp[1][0, 1]
                self.face['D'][0, 0], self.face['D'][0, 1] = cube_temp[2][0, 1], cube_temp[2][1, 1]
                self.face['L'][0, 1], self.face['L'][1, 1] = cube_temp[0][1, 0], cube_temp[0][1, 1]
                self.face['U'][1, 0], self.face['U'][1, 1] = cube_temp[3][0, 0], cube_temp[2][1, 0]

        elif face == 'B':
            self.face['B'] = np.rot90(self.face['B'], dir)
            if dir == -1:
                self.face['L'][0, 0], self.face['L'][1, 0] = cube_temp[0][0, 1], cube_temp[0][0, 0]
                self.face['D'][1, 1], self.face['D'][1, 0] = cube_temp[2][0, 0], cube_temp[2][1, 0]
                self.face['R'][0, 1], self.face['R'][1, 1] = cube_temp[1][0, 1], cube_temp[1][1, 1]
                self.face['U'][0, 1], self.face['U'][0, 0] = cube_temp[3][0, 1], cube_temp[3][1, 1]
            else:
                self.face['L'][0, 0], self.face['L'][1, 0] = cube_temp[1][1, 1], cube_temp[1][1, 0]
                self.face['D'][1, 1], self.face['D'][1, 0] = cube_temp[3][0, 1], cube_temp[3][1, 1]
                self.face['R'][0, 1], self.face['R'][1, 1] = cube_temp[0][1, 1], cube_temp[0][0, 1]
                self.face['U'][0, 1], self.face['U'][0, 0] = cube_temp[2][0, 0], cube_temp[2][1, 0]

        elif face == 'R':
            self.face['R'] = np.rot90(self.face['R'], dir)
            if dir == -1:
                self.face['B'][0, 0], self.face['B'][1, 0] = cube_temp[0][1, 1], cube_temp[0][0, 1]
                self.face['D'][0, 1], self.face['D'][1, 1] = cube_temp[5][1, 0], cube_temp[5][0, 0]
                self.face['F'][0, 1], self.face['F'][1, 1] = cube_temp[1][0, 1], cube_temp[1][1, 1]
                self.face['U'][1, 1], self.face['U'][0, 1] = cube_temp[4][1, 1], cube_temp[4][0, 1]
            else:
                self.face['B'][0, 0], self.face['B'][1, 0] = cube_temp[1][1, 1], cube_temp[1][0, 1]
                self.face['D'][0, 1], self.face['D'][1, 1] = cube_temp[4][0, 1], cube_temp[4][1, 1]
                self.face['F'][0, 1], self.face['F'][1, 1] = cube_temp[0][0, 1], cube_temp[0][1, 1]
                self.face['U'][1, 1], self.face['U'][0, 1] = cube_temp[5][0, 0], cube_temp[5][1, 0]

        elif face == 'L':
            self.face['L'] = np.rot90(self.face['L'], dir)
            if dir == -1:
                self.face['F'][0, 0], self.face['F'][1, 0] = cube_temp[0][0, 0], cube_temp[0][1, 0]
                self.face['D'][1, 0], self.face['D'][0, 0] = cube_temp[4][0, 0], cube_temp[4][1, 0]
                self.face['B'][0, 1], self.face['B'][1, 1] = cube_temp[1][1, 0], cube_temp[1][0, 0]
                self.face['U'][0, 0], self.face['U'][1, 0] = cube_temp[5][0, 1], cube_temp[5][1, 1]
            else:
                self.face['F'][0, 0], self.face['F'][1, 0] = cube_temp[1][0, 0], cube_temp[1][1, 0]
                self.face['D'][1, 0], self.face['D'][0, 0] = cube_temp[5][0, 1], cube_temp[5][1, 1]
                self.face['B'][0, 1], self.face['B'][1, 1] = cube_temp[0][1, 0], cube_temp[0][0, 0]
                self.face['U'][0, 0], self.face['U'][1, 0] = cube_temp[4][0, 0], cube_temp[4][1, 0]

        elif face == 'U':
            self.face['U'] = np.rot90(self.face['U'], dir)
            if dir == -1:
                self.face['R'][0, 0], self.face['R'][0, 1] = cube_temp[5][0, 0], cube_temp[5][0, 1]
                self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[3][0, 0], cube_temp[3][0, 1]
                self.face['L'][0, 0], self.face['L'][0, 1] = cube_temp[4][0, 0], cube_temp[4][0, 1]
                self.face['B'][0, 0], self.face['B'][0, 1] = cube_temp[2][0, 0], cube_temp[2][0, 1]
            else:
                self.face['R'][0, 0], self.face['R'][0, 1] = cube_temp[4][0, 0], cube_temp[4][0, 1]
                self.face['F'][0, 0], self.face['F'][0, 1] = cube_temp[2][0, 0], cube_temp[2][0, 1]
                self.face['L'][0, 0], self.face['L'][0, 1] = cube_temp[5][0, 0], cube_temp[5][0, 1]
                self.face['B'][0, 0], self.face['B'][0, 1] = cube_temp[3][0, 0], cube_temp[3][0, 1]

        elif face == 'D':
            self.face['D'] = np.rot90(self.face['D'], dir)
            if dir == -1:
                self.face['R'][1, 0], self.face['R'][1, 1] = cube_temp[4][1, 0], cube_temp[4][1, 1]
                self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[2][1, 0], cube_temp[2][1, 1]
                self.face['L'][1, 0], self.face['L'][1, 1] = cube_temp[5][1, 1], cube_temp[5][1, 0]
                self.face['B'][1, 0], self.face['B'][1, 1] = cube_temp[3][0, 1], cube_temp[3][1, 1]
            else:
                self.face['R'][1, 0], self.face['R'][1, 1] = cube_temp[5][1, 0], cube_temp[5][1, 1]
                self.face['F'][1, 0], self.face['F'][1, 1] = cube_temp[3][1, 0], cube_temp[3][1, 1]
                self.face['L'][1, 0], self.face['L'][1, 1] = cube_temp[4][1, 1], cube_temp[4][1, 0]
                self.face['B'][1, 0], self.face['B'][1, 1] = cube_temp[2][0, 1], cube_temp[2][1, 1]

        # U=0, D=1, L=2, R=3, F=4, B=5
        # Update cube array
        for i, face in enumerate(self.face):
            self.cube[i] = self.face[face]

    def __repr__(self) -> str:
        rep_string = ''
        for faces in self.face:
            rep_string += f'\n===={faces}====\n{self.face[faces]}'
        return rep_string


def column(matrix, i):
    return [row[i] for row in matrix]


if __name__ == "__main__":

    cube = Cube()
    # print(environment)
    cube.rotate_cube('D', 1)





