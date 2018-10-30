import unittest
import environment as env
import numpy as np


class TestEnv(unittest.TestCase):
    cube = env.Cube()

    def test_rotate_front(self):
        """
        Testing if the clockwise rotation is correct for the front side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'Front' rotation clockwise
        side = 'F'
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'w'], ['g', 'w']], dtype='<U1'),
                  np.array([['y', 'y'], ['g', 'g']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['b', 'b'], ['w', 'w']], dtype='<U1'),
                  np.array([['y', 'b'], ['y', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_front_counter(self):
        """
        Testing if the counter-clockwise rotation is correct for the front side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'Front' rotation counter-clockwise
        side = "F'"
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_left(self):
        """
        Testing if the clockwise rotation is correct for the left side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'Left' rotation clockwise
        side = 'L'
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['r', 'y'], ['r', 'y']], dtype='<U1'),
                  np.array([['y', 'o'], ['y', 'o']], dtype='<U1'),
                  np.array([['o', 'w'], ['o', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'w'], ['r', 'w']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_left_counter(self):
        """
        Testing if the counter-clockwise rotation is correct for the left side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'Left' rotation counter-clockwise
        side = "L'"
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_up(self):
        """
        Testing if the clockwise rotation is correct for the up side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'up' rotation clockwise
        side = 'U'
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['o', 'o'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['b', 'b'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['r', 'r'], ['b', 'b']], dtype='<U1'),
                  np.array([['g', 'g'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_up_counter(self):
        """
        Testing if the counter-clockwise rotation is correct for the up side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'up' rotation counter-clockwise
        side = "U'"
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_down(self):
        """
        Testing if the clockwise rotation is correct for the down side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'down' rotation clockwise
        side = 'D'
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['r', 'r']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['g', 'g']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['o', 'o']], dtype='<U1'),
                  np.array([['r', 'r'], ['b', 'b']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_down_counter(self):
        """
        Testing if the counter-clockwise rotation is correct for the down side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'down' rotation counter-clockwise
        side = "D'"
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_right(self):
        """
        Testing if the clockwise rotation is correct for the right side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'down' rotation clockwise
        side = 'R'
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'o'], ['y', 'o']], dtype='<U1'),
                  np.array([['o', 'w'], ['o', 'w']], dtype='<U1'),
                  np.array([['w', 'r'], ['w', 'r']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['y', 'r'], ['y', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_right_counter(self):
        """
        Testing if the counter-clockwise rotation is correct for the right side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'right' rotation counter-clockwise
        side = "R'"
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_back(self):
        """
        Testing if the clockwise rotation is correct for the back side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'back' rotation clockwise
        side = 'B'
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['y', 'g'], ['y', 'g']], dtype='<U1'),
                  np.array([['b', 'b'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['g', 'g']], dtype='<U1'),
                  np.array([['b', 'w'], ['b', 'w']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_rotate_back_counter(self):
        """
        Testing if the counter-clockwise rotation is correct for the back side

        """
        # Resetting the cube to initial position
        self.cube.reset_cube()

        # Testing 'back' rotation counter-clockwise
        side = "B'"
        rotation = list(self.cube.rotate_cube(side))
        result = [np.array([['g', 'g'], ['g', 'g']], dtype='<U1'),
                  np.array([['y', 'y'], ['y', 'y']], dtype='<U1'),
                  np.array([['o', 'o'], ['o', 'o']], dtype='<U1'),
                  np.array([['w', 'w'], ['w', 'w']], dtype='<U1'),
                  np.array([['b', 'b'], ['b', 'b']], dtype='<U1'),
                  np.array([['r', 'r'], ['r', 'r']], dtype='<U1')]

        np.testing.assert_array_equal(rotation, result)

    def test_stickers(self):
        """
        Testing if each side has only 4 stickers of each color.

        """
        rotation, _ = list(self.cube.scramble_cube(15))

        unique, counts = np.unique(rotation, return_counts=True)
        dictionary = dict(zip(unique, counts))

        self.assertEqual(all(value == 4 for value in dictionary.values()), True)


if __name__ == '__main__':
    unittest.main()
