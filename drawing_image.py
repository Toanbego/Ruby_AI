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
        rgb = (255, 255, 51)
    else:
        rgb = (0, 0, 0)
    return rgb


def image(pixel_x, pixel_y):
    # Initializing the image with RGB values and 8x6 pixels
    img = Image.new('RGB', (pixel_x, pixel_y))
    # Initializing a variable for drawing the image
    drawing = img.load()
    # Loops through every pixel
    for x in range(pixel_x):
        for y in range(pixel_y):
            if x == 2 or x == 3:
                # White
                if y == 0 or y == 1:
                    drawing[x, y] = (255, 255, 255)
                # Green
                elif y == 2 or y == 3:
                    drawing[x, y] = (0, 153, 0)
                # Blue
                elif y == 4 or y == 5:
                    drawing[x, y] = (37, 53, 199)
            # Orange
            elif x == 0 or x == 1:
                if y == 2 or y == 3:
                    drawing[x, y] = (255, 128, 0)
            # Red
            elif x == 4 or x == 5:
                if y == 2 or y == 3:
                    drawing[x, y] = (199, 37, 37)
            # Yellow
            elif x == 6 or x == 7:
                if y == 2 or y == 3:
                    drawing[x, y] = (255, 255, 51)
            # Black
            else:
                drawing[x, y] = (0, 0, 0)

            print(f'x:{x}, y:{y}')
        print('---------------------')
    # Saves the picture as a .PNG file
    img.save('rubix_cube.png', 'PNG')


# def image(pixel_x, pixel_y, stickers_color):
#     # Initializing the image with RGB values and 8x6 pixels
#     img = Image.new('RGB', (pixel_x, pixel_y))
#     # Initializing a variable for drawing the image
#     drawing = img.load()
#     # Loops through every pixel
#     for x in range(pixel_x):
#         for y in range(pixel_y):
#             if x == 2 or x == 3:
#                 # White
#                 if y == 0 or y == 1:
#                     drawing[x, y] = (255, 255, 255)
#                 # Green
#                 elif y == 2 or y == 3:
#                     drawing[x, y] = (0, 153, 0)
#                 # Blue
#                 elif y == 4 or y == 5:
#                     drawing[x, y] = (37, 53, 199)
#             # Orange
#             elif x == 0 or x == 1:
#                 if y == 2 or y == 3:
#                     drawing[x, y] = (255, 128, 0)
#             # Red
#             elif x == 4 or x == 5:
#                 if y == 2 or y == 3:
#                     drawing[x, y] = (199, 37, 37)
#             # Yellow
#             elif x == 6 or x == 7:
#                 if y == 2 or y == 3:
#                     drawing[x, y] = (255, 255, 51)
#             # Black
#             else:
#                 drawing[x, y] = (0, 0, 0)
#
#             print(f'x:{x}, y:{y}')
#         print('---------------------')
#     # Saves the picture as a .PNG file
#     img.save('rubix_cube.png', 'PNG')


if __name__ == '__main__':
    cube = np.array(
        [[['w', 'w'], ['w', 'w']], [['o', 'o'], ['o', 'o']], [['g', 'g'], ['g', 'g']], [['r', 'r'], ['r', 'r']],
         [['b', 'b'], ['b', 'b']], [['y', 'y'], ['y', 'y']]])
    image(8, 6, cube)
