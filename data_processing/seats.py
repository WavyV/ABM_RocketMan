import json
import os

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../data")


def get_student_locations(name):
    room = np.zeros((13, 14 + 6 + 1))
    with open(os.path.join(data_path, name)) as f:
        data = json.load(f)
    for student in data:
        location = student.get("seatlocation")
        i, j = list(map(int, location.split(",", 1)))
        room[i][j] = 1
    return room


if __name__ == "__main__":
    print(get_student_locations("fri-form.json"))
