import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from model import *

_compare_dict = {'lbp': 0, 'cluster': 1, 'entropy': 2}


"""
This is an adjusted version of Arran's 'utils.py'
Since the classroom_seating module now provides a methods that directly gives the desired binary model state,
The 'reduction' method is not needed anymore.

We should merge them into one generally applicable version.
"""




"""
Returns a list of counts for each length of each cluster of seated students.
This method escentially returns a histogram of group "lengths". Note an aisle
is considered the end of a group. Only counts horizontal groupings and so
assumes rows are independent.

Args:
    model_state: seating distribution as a binary matrix of seats
    aisles: A list of vertical aisles.

Returns:
    A numpy list of counts, where the ith element corresponds to the number of
    i lengthed groups, up to the max possible length defined by the aisles
"""
def count_clusters(model_state, aisles=[6]):
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
    model_state: seating distribution as a binary matrix of seats

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
    model_state: seating distribution as a binary matrix of seats

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
Compares two seating distributions by computing the Mean Square Error between their profiles.

Args:
    model_state_1: seating distribution as a binary matrix of seats
    model_state_2: seating distribution as a binary matrix of seats
    method: {'lbp', 'cluster', 'entropy'} The method to be used to compute the profiles
    aisles: If using the 'cluster' method, you can
        specify where the aisles are as a list. Default [0].

Returns:
    The MSE between the two profiles
"""
def compare(model_state_1, model_state_2, method='lbp', aisles=[0]):
    try:
        val = _compare_dict[method]

    except KeyError:
        raise ValueError("Method must be 'lbp', 'cluster', or 'entropy'.")

    if model_state_1.ndim != 2 or model_state_2.ndim != 2:
        raise ValueError("model must be 2D.")

    if method == 'lbp':
        # Use the Local Binary Method
        profile_1 = count_lbp(model_state_1)
        profile_2 = count_lbp(model_state_2)
        return calculate_mse(profile_1, profile_2)

    elif method == 'cluster':
        # Use the cluster size comparison
        profile_1 = count_clusters(model_state_1, aisles)
        profile_2 = count_clusters(model_state_2, aisles)
        return calculate_mse(profile_1, profile_2)

    elif method == 'entropy':
        # Use the entropy comparison
        profile_1 = get_entropy(model_state_1)
        profile_2 = get_entropy(model_state_2)
        return calculate_mse(profile_1, profile_2)

    else:
        return None
