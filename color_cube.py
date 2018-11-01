import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def color_plot(sticker):
    """
    Returns a value for each sticker
    :param sticker:
    :return:
    """
    if sticker == 'w':
        color = 0
    elif sticker == 'g':
        color = 1
    elif sticker == 'b':
        color = 2
    elif sticker == 'o':
        color = 3
    elif sticker == 'r':
        color = 4
    elif sticker == 'y':
        color = 5
    else:
        color = 6
    return color


def make_plot(sides, numeric):
    """
    Renders an image of the current state of the rubik's cube
    :param sides: The sides of the cube
    :param numeric: Numeric representation
    :return: None
    """
    # Simply inserting the black parts in the plot
    if numeric:
        popper = [[6, 6], [6, 6]]
    else:
        popper = [['x', 'x'], ['x', 'x']]

    sides = np.insert(sides, [0, 1, 4, 5, 5, 6], popper, axis=0)
    # For iterating over the image, 1337
    element = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2,
               3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5,
               6, 6, 7, 7, 8, 8, 6, 6, 7, 7, 8, 8,
               9, 9, 10, 10, 11, 11, 9, 9, 10, 10, 11, 11]
    el = 0
    # Initialize a zero matrix
    x_axis, y_axis = 8, 6
    plotting = np.empty(shape=(y_axis, x_axis))
    # Looping through all "pixels"
    for u in range(x_axis):
        for v in range(y_axis):

            # Giving values to the plotting matrix
            if numeric:
                plotting[v, u] = sides[element[el]][v % 2][u % 2]
            else:
                plotting[v, u] = color_plot(sides[element[el]][v % 2][u % 2])
            el += 1

    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'orange', 'red', 'yellow', 'black'])
    # Color boundaries
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Axes
    fig, ax = plt.subplots()
    ax.imshow(plotting, cmap=cmap, norm=norm)
    # draw grid-lines
    ax.grid(which='major', axis='both', linestyle='-', color='black', linewidth=2)
    ax.set_xticks(np.arange(0.5, 8, 1))
    ax.set_yticks(np.arange(0.5, 6, 1))
    # Show plot
    plt.show()


def main():
    cube = np.array(
        [[['o', 'o'], ['o', 'o']], [['w', 'w'], ['w', 'w']], [['g', 'g'], ['g', 'g']],
         [['b', 'b'], ['b', 'b']], [['r', 'r'], ['r', 'r']], [['y', 'y'], ['y', 'y']]])
    make_plot(cube)


if __name__ == '__main__':
    main()
