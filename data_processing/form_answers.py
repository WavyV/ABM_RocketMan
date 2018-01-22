import json
import os

import matplotlib.pyplot as plt
import numpy as np

"""
Function for extracting and plotting data from the questionnaire.

Functions are in the order they appear in the questionnaire.
"""

dir_path = os.path.dirname(os.path.realpath(__file__))
form_data_path = os.path.join(dir_path, "../data")

# Utility functions first #####################################################


def load(name):
    """Load data with given name."""
    with open(os.path.join(form_data_path, name)) as f:
        return json.load(f)


def get_int_answers(data, question_key, leq_5=False):
    """Return int data with given question key.

    If `leq_5` each int must be less than or equal to 5.
    """
    answers = map(lambda x: x.get(question_key), data)
    answers = filter(lambda x: x is not None and x != "", answers)
    answers = map(int, answers)
    if leq_5:
        answers = filter(lambda x: x <= 5, answers)
    return answers


def answers_and_plot(data, question_key, bins, plot=False, title=None,
                     leq_5=False):
    """Return a list of int data with given key.

    If `plot`, plot a histogram of the data.
    For `leq_5` see the function `get_int_answers`.
    """
    answers = get_int_answers(data, question_key, leq_5=leq_5)
    if plot:
        plt.hist(list(answers), bins=bins)
        if title:
            plt.title(title)
        plt.show()
    return answers


def get_seats(data, key, plot=False, title=None):
    """Get seats location matrix using the given question key.

    NOTE: aisles are not considered in the returned matrix.
    This function also returns the amount of missing entries.
    If `plot`, plot a histogram of the location matrix.
    """
    room = np.zeros((13, 14 + 6 + 1))
    missing_data = 0
    for student in data:
        location = student.get(key)
        if location:
            i, j = list(map(int, location.split(",", 1)))
            room[i][j] += 1
        else:
            missing_data += 1
    if plot:
        plt.imshow(room)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.show()
    return room, missing_data


# Questionnaire functions #####################################################


def crosscosts(data, plot=False):
    """Return a list of the crosscosts.

    If `plot`, plot a histogram of the returned data.
    """
    return answers_and_plot(data, "crosscost", bins=8, plot=plot,
                            title="Amount of crosscosts.")


def course_friends(data, plot=False):
    """Return a list of the amount of course friends people have.

    If `plot`, plot a histogram of the returned data.
    """
    return answers_and_plot(data, "coursefriends", bins=20, plot=plot,
                            title="Amount of course friends.")


def know_neighbour(data, plot=False):
    """Return a list for left data and a list for right data.

    If `plot`, plot a histogram of how well students know neighbours.
    """
    left = get_int_answers(data, "knowleft", leq_5=True)
    right = get_int_answers(data, "knowright", leq_5=True)
    both = list(left) + list(right)
    if plot:
        plt.hist(list(both), bins=6)
        plt.title("Familiarity to immediate neighbours.")
        plt.show()
    return left, right


def social_weighting(data, plot=False):
    """Return a list of the values of the social weighting slider.

    If `plot`, plot a histogram of the returned data.
    """
    return answers_and_plot(data, "slider", bins=20, plot=plot,
                            title="Sit next to friend / preferred seat. \n" +
                                  "Only want to sit next to a friend = 100.")


def importance_of_familiarity(data, plot=False):
    """Return a list of the importance of sitting next to someone familiar.

    If `plot`, plot a histogram of the returned data.
    """
    return answers_and_plot(data, "sitnexttofamiliar", bins=5, plot=plot,
                            title="Importance of sitting next to someone " +
                                  "familiar.", leq_5=True)


def importance_of_person(data, plot=False):
    """Return a list of the importance of sitting next to a person.

    If `plot`, plot a histogram of the returned data.
    """
    return answers_and_plot(data, "sitnexttoperson", bins=5, plot=plot,
                            title="Importance of sitting next to a person.",
                            leq_5=True)


def get_actual_seats(data, plot=False):
    """Returns a matrix of where students said they sat.

    If `plot`, plot returned data using imshow.
    """
    return get_seats(data, "seatlocation", plot=plot,
                     title="Where students are sitting.")[0]


def get_preferred_seats__some_unavailable(data, plot=False):
    """Returns a matrix of students preferred seats.

    From the question where some seats are unavailable (black).
    If `plot`, plot returned data using imshow.
    """
    return get_seats(data, "seatpreffered", plot=plot,
                     title="Where students would prefer to sit. \n" +
                           "For question with unavailable (black) seats.")[0]


if __name__ == "__main__":
    data = load("fri-form.json")
    crosscosts(data, plot=True)
    course_friends(data, plot=True)
    know_neighbour(data, plot=True)
    social_weighting(data, plot=True)
    importance_of_familiarity(data, plot=True)
    importance_of_person(data, plot=True)
    get_actual_seats(data, plot=True)
    get_preferred_seats__some_unavailable(data, plot=True)
