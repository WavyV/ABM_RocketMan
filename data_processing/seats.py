import json
import os

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../data")


def get_seats(name, key):
    """Get seats location matrix using the given question key.

    Also returns the amount of missing entries.
    """
    room = np.zeros((13, 14 + 6 + 1))
    with open(os.path.join(data_path, name)) as f:
        data = json.load(f)
    missing_data = 0
    for student in data:
        location = student.get(key)
        if location:
            i, j = list(map(int, location.split(",", 1)))
            room[i][j] = 1
        else:
            missing_data += 1
    return room, missing_data


def get_student_locations(name):
    return get_seats(name, "seatlocation")[0]


def get_preferred_seats(name):
    return get_seats(name, "seatpreffered")[0]


if __name__ == "__main__":
    print(get_student_locations("fri-form.json"))
    print(get_preferred_seats("fri-form.json"))
