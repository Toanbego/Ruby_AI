from PIL import Image
import numpy as np


def color(sticker):
    if sticker == 'w':
        rgb = (255, 255, 255)
    elif sticker == 'g':
        rgb = (0, 153, 0)
    elif sticker == 'b':
        rgb = (37, 53, 199)
    elif sticker == 'o':
        rgb = (255, 128, 0)
    elif sticker == 'r':
        rgb = (199, 37, 37)
    elif sticker == 'y':
        rgb = (255, 255, 51)
    else:
        rgb = (0, 0, 0)
    return rgb


def make_image(sides):
    x_axis, y_axis = 8, 6
    # Initializing the image with RGB values and 8x6 pixels
    img = Image.new('RGB', (x_axis, y_axis))
    # Initializing a variable for drawing the image
    drawing = img.load()
    element = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2,
               3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5,
               6, 6, 7, 7, 8, 8, 6, 6, 7, 7, 8, 8,
               9, 9, 10, 10, 11, 11, 9, 9, 10, 10, 11, 11]
    el = 0
    for u in range(x_axis):
        for v in range(y_axis):
            drawing[u, v] = color(sides[element[el]][v % 2][u % 2])
            el += 1
    img.save('rubix_cube.png', 'PNG')


def main():
    cube = np.array(
        [[['x', 'x'], ['x', 'x']], [['o', 'o'], ['o', 'o']], [['x', 'x'], ['x', 'x']],
         [['w', 'w'], ['w', 'w']], [['g', 'g'], ['g', 'g']], [['b', 'b'], ['b', 'b']],
         [['x', 'x'], ['x', 'x']], [['r', 'r'], ['r', 'r']], [['x', 'x'], ['x', 'x']],
         [['x', 'x'], ['x', 'x']], [['y', 'y'], ['y', 'y']], [['x', 'x'], ['x', 'x']]])
    make_image(cube)


if __name__ == '__main__':
    main()
