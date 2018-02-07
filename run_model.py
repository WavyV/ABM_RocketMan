import sys
from os import path
import pickle
from model import *
from data_processing import process_form

MODEL_DATA_PATH = "animation_data"
FILE_NAME = "model_data.json"
NUM_ITERATIONS = 150
CLASS_SIZE = 150
# position + friendship + sociability + accessability
MODEL_COEFS = [[0,0,0,1], [0,0,1,0], [0,1,0,1]]
BINS = [0, 0.1, 0.5]

MODEL_INPUT_PATH = "model_input"
DEFAULT_DEG_SEQ = "_degree_sequence.pkl"
DEFAULT_SOC_SEQ = "_sociability_sequence.pkl"
DEFAULT_POS_UTIL = "pos_utilities.pkl"
DEFAULT_POS_UTIL_BINS = "pos_utility_bins.pkl"

"""
This module provides methods to run simulations of the ClassroomModel
If used as a script: generates the model data that is needed for animation

Usage:
    python3 run_model.py [file_name_for_animation_data]
"""


def get_default_pos_utilities():

    file_path = path.join(MODEL_INPUT_PATH, DEFAULT_POS_UTIL)
    if path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    else:
        pos_utilities = process_form.seat_location_scores().T
        with open(file_path, 'wb') as f:
            pickle.dump(pos_utilities, f)
        return pos_utilities

def get_default_pos_utility_bins():

    file_path = path.join(MODEL_INPUT_PATH, DEFAULT_POS_UTIL_BINS)
    if path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    else:
        pos_utilities = process_form.seat_location_bins(BINS).T
        with open(file_path, 'wb') as f:
            pickle.dump(pos_utilities, f)
        print(pos_utilities)
        return pos_utilities

def get_block_pos_utilities():

    # Assumes classroom to be default shape
    seating_bins = np.ones((14, 22))

    # first three rows very undisirable
    seating_bins[0:3, 7:] *= 0.05
    seating_bins[0:3, 0:7] *= 0.0

    # back five rows somewhat desirable
    seating_bins[-5:, 7:] *= 0.5
    seating_bins[-5:, 0:7] *= 0.2

    # middle section to the left also quite desirable
    seating_bins[3:-5, 0:7] *= 0.7

    return seating_bins.T

def get_default_sociability_sequence(class_size):

    file_path = path.join(MODEL_INPUT_PATH, "size_" + str(class_size) + DEFAULT_SOC_SEQ)
    if path.isfile(file_path):
        with open(file_path, 'rb') as f:
            s = pickle.load(f)
            #print("sociability sequence properties:")
            #print("mean: {}".format(np.mean(s)))
            return s

    else:
        sociability_generator = process_form.agent_sociability_gen()
        sociability_sequence = [next(sociability_generator) for _ in range(class_size)]
        with open(file_path, 'wb') as f:
            pickle.dump(sociability_sequence, f)
        return sociability_sequence

def generate_sociability_sequence(class_size, distribution, seed=0):

    np.random.seed(seed)

    if distribution == "cauchy":
        sociability_sequence = np.random.standard_cauchy(size=class_size)
    elif distribution == "gaussian":
        sociability_sequence = np.random.normal(size=class_size)
    elif distribution == "uniform_aversion":
        sociability_sequence = np.random.uniform(-1,1, size=class_size)
    elif distribution == "uniform_affection":
        sociability_sequence = np.random.uniform(-1,1, size=class_size)
    else:
        print("Wrong distribution given! Ending program.")
        quit()

    return sociability_sequence

def get_default_degree_sequence(class_size):

    file_path = path.join(MODEL_INPUT_PATH, "size_" + str(class_size) + DEFAULT_DEG_SEQ)
    if path.isfile(file_path):
        with open(file_path, 'rb') as f:
            s = pickle.load(f)
            #print("degree sequence properties:")
            #print("mean: {}".format(np.mean(s)))
            #print("min: {}".format(np.min(s)))
            #print("max: {}".format(np.max(s)))
            #print("len: {}".format(len(s)))
            return s

    else:
        friendship_generator = process_form.agent_friends_gen()
        degree_sequence = [next(friendship_generator) for _ in range(class_size)]
        with open(file_path, 'wb') as f:
            pickle.dump(degree_sequence, f)
        return degree_sequence


"""
Initialize the default ClassroomModel (based on collected data)

Args:
    coefs: coefficients for the utility function
    class_size: number of students in the class, forming the social network
    seed: for random number generation
    seat_fraction: fraction of available seats to be considered for seat choice
    deterministic_choice: if True, students always choose one of the seats with highest total utility.
                        Otherwise, the decision is made probabilisticly, converting seat utilities into probabilities.
    social_aversion: if True, an artificial sociability sequence is generated that includes negative sociability values representing social aversion.
                        Otherwise, the sociability sequence derived from the data is used, where sociability values only range from 0 to 1.

Returns:
    model: the created model instance
"""
def init_default_model(coefs, class_size, seed=0, seat_fraction=0.5,
                       deterministic_choice=True, social_aversion=False,
                       scale=True):

    # Using the default classroom size of [6,14,0] blocks and 14 rows

    """ run the following to use the entire range of pos_utilities """
    #classroom = ClassroomDesign(pos_utilities=get_default_pos_utilities())

    """ run the following to use bins of pos_utilities """
    # classroom = ClassroomDesign(pos_utilities=get_default_pos_utility_bins())

    """ run the following to use custom bins of pos_utilities """
    classroom = ClassroomDesign(pos_utilities=get_block_pos_utilities())

    # The degree sequence for the social network is sampled from the observed
    # distribution of number of friends
    degree_sequence = get_default_degree_sequence(class_size)

    if social_aversion:
        # The sociability sequence is sampled randomly from a distribution, including negative values
        """ use the following line to sample from a gaussian distribution"""
        #sociability_sequence = generate_sociability_sequence(class_size, "gaussian", seed)

        """ use the following line to sample from a cauchy distribution"""
        #sociability_sequence = generate_sociability_sequence(class_size, "cauchy", seed)

        """ use the following line to sample from a uniform distribution in the interval [-1,1]"""
        #sociability_sequence = generate_sociability_sequence(class_size, "uniform_aversion", seed)

        """ use the following line to sample from a uniform distribution in the interval [0,1]"""
        sociability_sequence = generate_sociability_sequence(class_size, "uniform_affection", seed)

        # scale to range [-1,1]
        sociability_sequence = (((sociability_sequence - np.min(sociability_sequence)) * 2) / (np.max(sociability_sequence - np.min(sociability_sequence)))) - 1

    else:
        # The sociability sequence is based on the distribution observed in the collected data
        sociability_sequence = get_default_sociability_sequence(class_size)

    # create the model
    model = ClassroomModel(classroom, coefs,
                           sociability_sequence=sociability_sequence,
                           degree_sequence=degree_sequence, seed=seed,
                           seat_fraction=seat_fraction,
                           deterministic_choice=deterministic_choice,
                           scale=scale)

    return model


"""
Determine relevant properties of the current model state and create an image representation

Args:
    model: the classroom model to analyse
    utilities: list of booleans [location, sociability, friendship] specifying
        which utilities are used in the model

Returns:
    image: 2D array representing the classroom.
                - The level of seat blocking is represented by decreasing negative values (the smaller the costlier to reach the seat)
                - The level of "happiness" of each student is represented by positive values (the larger the happier)
    info: 3D array comprising the 4 matrices 'seat_utilities', 'student_IDs', 'student_sociabilities' and 'initial_happiness'
"""
def get_model_state(model):
    image = -0.8*np.ones((model.classroom.num_rows, model.classroom.width))
    info = np.zeros((model.classroom.num_rows, model.classroom.width, 4))

    for seat in model.seats.flat:
        if type(seat) != Seat:
            continue

        x, y = seat.pos

        info[y,x,0] = model.classroom.pos_utilities[x,y]
        if seat.student is None:
            # seat is available. Determine level of accessibility
            image[y,x] = -2 + seat.accessibility
        else:
            # seat is occupied. Set value based on the student's happiness
            image[y,x] = 1
            image[y,x] += seat.get_happiness(seat.student)

            # save student's properties
            info[y,x,1] = seat.student.unique_id
            info[y,x,2] = seat.student.sociability
            info[y,x,3] = seat.student.initial_happiness


    for pos in model.classroom.entrances:
        image[pos[1],pos[0]] = -3

    return image, info


def generate_model_names():

    model_names = []
    for m in MODEL_COEFS:
        name = []
        for i,c in enumerate(m):
            if i==0 and c>0:
                name.append("position")
            if i==1 and c>0:
                name.append("friendship")
            if i==2 and c>0:
                name.append("sociability")
            if i==3 and c>0:
                name.append("accessibility")
        model_names.append(" + ".join(name))
    return model_names


"""
Generate the necessary model states for an animation.
"""
def generate_data(models, num_iterations, data_path=None):
    # Save model state for each model for each step
    all_model_states = []
    for i in range(len(models)):
        print("Model {0}".format(i + 1))
        model_states = []
        for iteration in range(num_iterations):
            print("Iteration {0}".format(iteration + 1))
            models[i].step()
            model_states.append(get_model_state(models[i]))
        all_model_states.append(model_states)

    model_names = generate_model_names()
    data = [MODEL_COEFS, model_names, all_model_states]

    # Save all model state data.
    if data_path is None:
        data_path = path.join(MODEL_DATA_PATH, FILE_NAME)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def final_model(model, num_iterations):
    """Return the final model state after running for given iterations."""
    for _ in range(num_iterations):
        model.step()
    return model


if __name__ == "__main__":

    # initialize and run models with given utility coefficients [position, friendship, sociability, accessibility]
    models = []
    for coefs in MODEL_COEFS:
        """ use the following to simulate the model with probabilistic choice among the best seat_fraction percent """
        #models.append(init_default_model(coefs, CLASS_SIZE, seat_fraction=0.1, deterministic_choice=False))

        """ use the following to simulate the model with deterministic choice of the highest rated seat """
        models.append(init_default_model(coefs, CLASS_SIZE, deterministic_choice=True, social_aversion=True))

    try:
        generate_data(models, NUM_ITERATIONS, path.join(MODEL_DATA_PATH, sys.argv[1]))
    except:
        generate_data(models, NUM_ITERATIONS)
