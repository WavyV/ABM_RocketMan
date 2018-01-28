import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from classroom_seating import *

_compare_dict = {'lbp': 0, 'cluster': 1, 'entropy': 2}


"""
############################################
Essential tools for analyzing the model data
############################################
"""

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
    image = -8*np.ones((model.grid.height, model.grid.width))
    info = np.zeros((model.grid.height, model.grid.width, 4))

    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat:
                # save seat utility
                info[y,x,0] = model.classroom.pos_utilities[x,y]
                if agent.student is None:
                    # seat is available. Determine level of blocking
                    image[y,x] = -10 + agent.get_accessibility()
                else:
                    # seat is occupied. Set value based on the student's happiness
                    image[y,x] = 10
                    u_friendship, u_sociability = agent.get_social_utility(agent.student)
                    image[y,x] += agent.get_position_utility() + u_friendship + u_sociability

                    # save student's properties
                    info[y,x,1] = agent.student.unique_id
                    info[y,x,2] = agent.student.sociability
                    info[y,x,3] = agent.student.initial_happiness

    for pos in model.classroom.entrances:
        image[pos[1],pos[0]] = -20

    return image, info

"""
Run model to completion and save locally the final state.

Args:
    model: The ClassroomModel instance to run and save.
    dir: Directory and filename.pkl to save final model state to.
    state: Save just the final state of the model as an array instead of the full model instance. Default False.
"""
def run_model(model, dir="./model.pkl", state=False):
    for i in range(model.max_num_agents):
        model.step()

    model_state = get_model_state(model)

    with open(path, "wb") as f:
        if state:
            pkl.dump(model_state, f)
        else:
            pkl.dump(model, f)


"""
Reduces a model state to a binary matrix where 1 represents a taken seat, 0
otherwise. Also removes all aisles. Useful for analysis.

Args:
    model_state: Model state to reduce.

Returns:
    A 2D numpy array representing taken seats.
"""
def reduce_model_state(model):
    reduced_state = get_model_state(model)[0]

    # Remove all the aisle columns
    for aisle in reversed(model.classroom.aisles_x):
        reduced_state = np.delete(reduced_state, aisle, 1)

    # Remove any horizontal aisles
    for aisle in reversed(model.classroom.aisles_y):
        reduced_state = np.delete(reduced_state, aisle, 0)

    # Set 1 for taken seats, and 0 otherwise
    for seat in np.nditer(reduced_state, op_flags=['readwrite']):
        seat[...] = 1 if (seat > 0) else 0

    return reduced_state


"""
Returns a list of counts for each length of each cluster of seated students.

This method escentially returns a histogram of group "lengths". Note an aisle
is considered the end of a group. Only counts horizontal groupings and so
assumes rows are independent.

Args:
    model_state: The reduced model state (use reduce_model_state() before).
    aisles: A list of vertical aisles.

Returns:
    A numpy list of counts, where the ith element corresponds to the number of
    i lengthed groups, up to the max possible length defined by the aisles
"""
def count_clusters(model_state, aisles):
    # where the final counts will be stored
    counts = np.zeros(model_state.shape[1] + 1)

    for row in model_state:
        # split the row into blocks and iterate over the seats
        for block in np.split(row, aisles):
            c = 0
            for seat in block:
                if seat == 1:
                    c += 1
                else:
                    counts[c] += 1
                    c = 0
            counts[c] += 1

    # the value at i = 0 will be non-sensical, so set to 0
    counts[0] = 0

    return counts


"""
Returns a count of each value for a Local Binary Pattern (LBP) over all seats.

This method escentially returns a histogram that aims to capture

Advantages:
    - Captures somewhat the spacial distribution of seating,
    - The fine grain allows distinction between very similar but different models,
    - Can be used to compare any sized lecture theater.

Args:
    model_state: The reduced model state.

Returns:
    A numpy list of length 256, with each element corresponding to the count of
    each uniquley defined LBP
"""
def count_lbp(model_state):
    # the relative coordinates in sequence in order to traverse around the
    # seat so to build up the binary representation of the seat's 8 neighbors
    i_deltas = [-1, -1, -1, 0, 1, 1, 1, 0]
    j_deltas = [-1, 0, 1, 1, 1, 0, -1, -1]


    # where the final counts will be stored
    counts = np.zeros(256)

    # iterate over all seats except the outer edges
    for i in np.arange(1, model_state.shape[0] - 1):
        for j in np.arange(1, model_state.shape[1] - 1):
            # decimal representation of surrounding seats
            dec = 0
            for k, (i_d, j_d) in enumerate(zip(i_deltas, j_deltas)):
                dec += model_state[i+i_d][j+j_d] * 2**k

            # increase the count for this LBP
            counts[int(dec)] += 1

    return counts


"""
Calculates the entropy profile of a model state.

Args:
    model_state: A reduced model state.

Returns:
    A list of entropies for neighborhood sizes k = 1 to minimium matrix side
    length.
"""
def get_entropy(model_state):
    entropies = []
    height, width = model_state.shape

    for k in range(1, min(height, width) + 1):
        matrix_k = np.zeros([height - k + 1, width - k + 1])

        # slide k by k window over model state and take the mean
        for (row, col), val in np.ndenumerate(model_state):
            if not ((row > (height - k)) or (col > (width - k))):
                sub_matrix = np.zeros((k, k))

                for i in range(k):
                    for j in range(k):
                        sub_matrix[i, j] = model_state[row+i, col+j]

                matrix_k[row, col] = sub_matrix.mean()

        # calculate entropy for this sub matrix by generating a discrete
        # distribution of values and their frequencies
        unique, counts = np.unique(matrix_k, return_counts=True)
        total = sum(counts)
        dist = np.array([x / total for x in counts])
        entropy = -sum(dist * np.log2(dist))
        entropies.append(entropy)

    return entropies


"""
############################################
Analysis Methods
############################################
"""

"""
Returns the Mean Square Error between two lists of numbers. If both lists are
different length, only compares up to the length of the shortest list.

Args:
    list1: The first list.
    list2: The second list.

Returns:
    The MSE between the two lists.
"""
def calculate_mse(list1, list2):
    mse = 0
    for x, y in zip(list1, list2):
        mse += (x - y)**2
    return mse / min(len(list1), len(list2))


"""
Compares a model with an expected profile and returns the Mean Square Error.

Args:
    model: Model to be compared. Can be an instance of a ClassroomModel or a
        numpy matrix of seating positions.
    profile: The profile to be used.
    method: {'lbp', 'cluster', 'entropy'} The method the profile is of.
    aisles: If the model just a matrix, and using the 'cluster' method, you can
        specify where the aisles are as a list. Default [0].

Returns:
    The MSE between the model and the profile using the choice of method
"""
def compare_model(model, profile, method='lbp', aisles=[0]):
    try:
        val = _compare_dict[method]

    except KeyError:
        raise ValueError("Method must be 'lbp', 'cluster', or 'entropy'.")

    # if a ClassroomModel is passed in, reduce this model first
    if isinstance(model, ClassroomModel):
        model_state = reduce_model_state(model)
        _aisles = model.classroom.aisles_x
    else:
        model_state = np.asarray(model)
        if model_state.ndim != 2:
            raise ValueError("model must be 2D.")
        _aisles = aisles


    if method == 'lbp':
        # Use the Local Binary Method
        model_profile = count_lbp(model_state)
        return calculate_mse(model_profile, profile)

    elif method == 'cluster':
        # Use the cluster size comparison
        model_profile = count_clusters(model_state, _aisles)
        return calculate_mse(model_profile, profile)

    elif method == 'entropy':
        # Use the entropy comparison
        model_profile = get_entropy(model_state)
        return calculate_mse(model_profile, profile)

    else:
        return None


"""
Generates a profile of a set of models with selected method.

Args:
    models: A list of models to analyse. Assumes all be the same shape.
    method: {'lbp', 'cluster', 'entropy'} The method to use.

Returns:
    A list of counts that is the average profile of the given models.

"""
def generate_profile(models, method='lbp'):
    try:
        val = _compare_dict[method]

    except KeyError:
        raise ValueError("Method must be 'lbp', 'cluster', or 'entropy'.")

    # reduce all the models first
    reduced_models = []
    for m in models:
        reduced_models.append(reduce_model_state(m))

    # setup profile depending on method type
    if method == 'lbp':
        profile = np.zeros(256)
        f = count_lbp
        args = []
    elif method == 'cluster':
        profile = np.zeros(reduced_models[0].shape[1]+1)
        f = count_clusters
        args = (models[0].classroom.aisles_x)
    elif method == 'entropy':
        profile = np.zeros(min(reduced_models[0].shape))
        f = get_entropy
        args = []

    # build profile
    for rm in reduced_models:
        if method == 'cluster':
            profile += f(rm, args)
        else:
            profile += f(rm)

    return profile / len(models)

"""

"""
def plot_model_profile():
    pass
