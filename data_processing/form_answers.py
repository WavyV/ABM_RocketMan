import json
import os

import matplotlib.pyplot as plt
import numpy as np

import time

"""
Function for extracting and plotting data from the questionnaire.
Functions are in the order they appear in the questionnaire.
Usage as a script to visualize a form's responses:
    python3 form_answers.py fri-form.json
"""

dir_path = os.path.dirname(os.path.realpath(__file__))
form_data_path = os.path.join(dir_path, "../data")

# Utility functions first #####################################################


def load(name):
    """Load data with given name."""
    with open(os.path.join(form_data_path, name)) as f:
        return json.load(f)


def get_int_answers(data, question_key, leq_5=False):
    """Return a list of int data with given question key.
    If `leq_5` each int must be less than or equal to 5.
    """
    answers = map(lambda x: x.get(question_key), data)
    answers = filter(lambda x: x is not None and x != "", answers)
    answers = map(int, answers)
    if leq_5:
        answers = filter(lambda x: x <= 5, answers)
    return list(answers)


def answers_and_hist(data, question_key, bins="auto", plot=False, title=None,
                     leq_5=False):
    """Return a list of int data with given key.
    If `plot`, plot a histogram of the data.
    For `leq_5` see the function `get_int_answers`.
    """
    answers = get_int_answers(data, question_key, leq_5=leq_5)
    hist = np.histogram(answers, bins=bins)
    if plot:
        plt.hist(answers, bins=bins)
        if title:
            plt.title(title)
        #plt.savefig(question_key + ".png", bbox_inches='tight')
        plt.show()
    return answers, hist


def get_seats(data, key, plot=False, title=None):
    """Get seats location matrix using the given question key.
    NOTE: aisles are not considered in the returned matrix.
    If `plot`, plot a histogram of the location matrix.
    """
    room = np.zeros((13, 14 + 6 + 1))
    for student in data:
        location = student.get(key)
        if location:
            i, j = list(map(int, location.split(",", 1)))
            room[i][j] += 1
    aisle_index = 6
    room = np.delete(room, aisle_index, 1)
    if plot:
        plt.imshow(room)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.savefig(key + ".png", bbox_inches='tight')
        plt.show()
    return room


# Questionnaire functions #####################################################


def crosscosts(data, plot=False):
    """Return a list of the crosscosts, and the histogram data.

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "crosscost", bins=8, plot=plot,
                            title="Amount of crosscosts.")


def course_friends(data, plot=False):
    """Return the amount of course friends people have, and the histogram data.

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "coursefriends", bins="auto", plot=plot,
                            title="Amount of course friends.")


def know_neighbour(data, plot=False):
    """Return a list for left data and a list for right data.
    If `plot`, plot a histogram of how well students know neighbours.
    """
    left = get_int_answers(data, "knowleft", leq_5=True)
    right = get_int_answers(data, "knowright", leq_5=True)
    if plot:
        plt.hist(left + right, bins=6)
        plt.title("Familiarity to immediate neighbours.")
        plt.show()
    return left, right


def social_weighting(data, plot=False):
    """Return a list of the social slider values, and the histogram data.

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "slider", bins=20, plot=plot,
                            title="Sit next to friend / preferred seat. \n" +
                                  "Only want to sit next to a friend = 100.")


def importance_of_familiarity(data, plot=False):
    """Return a list of the importance of sitting next to someone familiar, and the
    histogram data.

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "sitnexttofamiliar", bins=5, plot=plot,
                            title="Importance of sitting next to someone " +
                                  "familiar.", leq_5=True)


def importance_of_person(data, plot=False):
    """Return a list of the importance of sitting next to a person, and the
       histogram data.

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "sitnexttoperson", bins=5, plot=plot,
                            title="Importance of sitting next to a person.",
                            leq_5=True)


def importance_of_location(data, plot=False):
    """Return a list of the importance of sitting in preferred location, and the
    histogram data.

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "sitgoodlocation", bins=5, plot=plot,
                            title="Importance of sitting in preferred seat.",
                            leq_5=True)


def importance_of_accessibility(data, plot=False):
    """Return a list of the importance of sitting in easy to reach seat, and
    the histogram data

    If `plot`, plot the histogram of the returned data.
    """
    return answers_and_hist(data, "siteasyreach", bins=5, plot=plot,
                            title="Importance of sitting in accessible seat.",
                            leq_5=True)


def get_actual_seats(data, plot=False):
    """Returns a matrix of where students said they sat.
    If `plot`, plot returned data using imshow.
    """
    return get_seats(data, "seatlocation", plot=plot,
                     title="Where students are sitting.")


def get_preferred_seats(data, plot=False):
    """Returns a matrix of students preferred seats.

    If `plot`, plot returned data using imshow.
    """
    return get_seats(data, "seatpreffered", plot=plot,
                     title="Where students would prefer to sit.")


FILENAMES = ["fri-form.json", "wed_24.json"]
DATA = list(map(load, FILENAMES))
ALL_DATA = list(sum(DATA, []))

if __name__ == "__main__":
    crosscosts(ALL_DATA, plot=True)
    answers, hist = course_friends(ALL_DATA, plot=True)
    know_neighbour(ALL_DATA, plot=True)
    social_weighting(ALL_DATA, plot=True)
    importance_of_familiarity(ALL_DATA, plot=True)
    importance_of_person(ALL_DATA, plot=True)
    importance_of_location(ALL_DATA, plot=True)
    importance_of_accessibility(ALL_DATA, plot=True)
    get_actual_seats(ALL_DATA, plot=True)
    get_preferred_seats(ALL_DATA, plot=True)
