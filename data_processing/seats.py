import json
import os

import matplotlib.pyplot as plt
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../data")


def load(name):
    """Load data with given name."""
    with open(os.path.join(data_path, name)) as f:
        return json.load(f)


def get_seats(data, key):
    """Get seats location matrix using the given question key.

    Also returns the amount of missing entries.
    """
    room = np.zeros((13, 14 + 6 + 1))
    missing_data = 0
    for student in data:
        location = student.get(key)
        if location:
            i, j = list(map(int, location.split(",", 1)))
            room[i][j] = 1
        else:
            missing_data += 1
    return room, missing_data


def get_student_locations(data):
    return get_seats(data, "seatlocation")[0]


def get_preferred_seats(data):
    return get_seats(data, "seatpreffered")[0]


def social_weighting(data):
    weightings = map(lambda x: x.get("slider"), data)
    weightings = filter(lambda x: x is not None, weightings)
    weightings = map(int, weightings)
    plt.hist(list(weightings), bins=100)
    plt.show()


def know_neighbour(data):
    left = map(lambda x: x.get("knowleft"), data)
    right = map(lambda x: x.get("knowright"), data)
    both = list(left) + list(right)
    both = map(lambda x: int(x) if isinstance(x, str) else x, both)
    both = filter(lambda x: x <= 5, both)
    plt.hist(list(both), bins=6)
    plt.show()


if __name__ == "__main__":
    data = load("fri-form.json")
    print(get_student_locations(data))
    print(get_preferred_seats(data))
    # social_weighting(data)
    know_neighbour(data)
