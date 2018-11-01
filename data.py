import pandas as pd
import configparser
import environment
import itertools

config = configparser.ConfigParser()
config.read("config.ini")

def generate_deterministic_data(cube, generate_deterministic):
    """
    Generates a deterministic data set
    :param cube:
    :param generate_deterministic: If true: continue, If false: return
    :return:
    """
    if generate_deterministic is False:
        return

    # Initialize variable
    nodes = config['dataset'].getint('nodes')
    file_name = config['dataset']['write_file_name']
    data_set = []
    list_of_permutations = []

    # Permute the number of possible combinations up to max 10 permutations
    noe = 0
    for node in range(1, nodes+1):
        list_of_permutations.append(list(itertools.permutations(cube.action_space, node)))
        noe += len(list_of_permutations[node-1])

    # Make a search tree
    for permutations in list_of_permutations:  # Loop through action sequence
        for action_seq in permutations:
            cube.cube, cube.face = cube.reset_cube()  # Reset the cube each time
            for action in action_seq:

                cube.rotate_cube(action)  # Perform actions
            data_set.append((cube.cube, action_seq))
    print("Finished with permutations")
    # # Create Data frame and write to .pkl
    df = pd.DataFrame(data_set, columns={"Cube": data_set[:][0], "Actions": data_set[:][1]})
    print("Writing dataset")
    df.to_pickle(f"data/{file_name}.pkl")


def generate_training_samples(cube, generate_random, scrambles=1):
    """
    Generate the training set for supervised learning
    :param cube:
    :param generate_random:
    :param scrambles:
    :return:
    """
    if generate_random is False:
        return

    data_set = []   # Will consist of the cube array
    data_set_size = config['dataset'].getint('sample_size')
    file_name = config['dataset']['write_file_name']
    print("Scrambling cubes")
    # Generate randomly scrambled cubes
    for sample in range(data_set_size):

        # Reset the cube each time
        cube.cube, cube.face = cube.reset_cube()

        # Append the scrambled cube, with the actions it took to get there
        data_set.append(cube.scramble_cube(scrambles))
    print("Writing to file")
    # Create Data frame and write to .pkl
    df = pd.DataFrame(data_set, columns={"Cube": data_set[:][0], "Actions": data_set[:][1]})
    df.to_pickle(f"data/{file_name}.pkl")


def read_data_set(file_name):
    """
    Reads a .pkl file an returns the columns separated into cubes
    and actions
    :param file_name:
    :return:
    """
    # Read is activated or not
    if config['dataset'].getboolean('read_data') is False:
        return
    df = pd.read_pickle(f"data/{file_name}.pkl")
    return df, df["Cube"], df["Actions"]

if __name__ == "__main__":
    rubiks_cube = environment.Cube()
    # Generate Training set if set to True
    generate_deterministic_data(rubiks_cube, config['dataset'].getboolean('generate_deterministic'))
    generate_training_samples(rubiks_cube, config['dataset'].getboolean('generate_random'), scrambles=2)

    # Read data
    data, data_cube, data_actions = read_data_set(config['dataset']['read_file_name'])
    print(data_cube, data_actions)