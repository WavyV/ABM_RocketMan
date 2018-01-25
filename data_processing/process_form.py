import json
import sys

import matplotlib.pyplot as plt
import numpy as np

import form_answers
import taken_seats

"""
Process form answers into useful information for the model.

Form answers are calculated using the form_answers module.
"""


def get_seat_stats(seats, max_radius, taken=None, center=(3, 12)):
    """Return the total amount of seat choices at each radius / the amount of seats
       at that radius.

       NOTE: Use `seat_radii_prob` as a convenience wrapper.
    """
    c_i, c_j = center
    radii = list(range(max_radius))

    def _get_seats_stats(radius):
        choices_in_radius = 0
        total_seats = 0
        # Consider seats d_j away from center.
        # We use `set` to filter duplicates in-case radius is 0.
        for j in set([c_j + radius, c_j - radius]):
            for d_i in range(-radius, radius + 1):
                i = c_i + d_i
                exclude_seat = taken is not None and taken[i][j]
                if not exclude_seat:
                    choices_in_radius += seats[i][j]
                    total_seats += 1
        # Consider seats d_i away from center.
        # We need to avoid the corners already counted.
        for i in set([c_i + radius, c_i - radius]):
            for d_j in range(-(radius - 1), radius):
                j = c_j + d_j
                exclude_seat = taken is not None and taken[i][j]
                if not exclude_seat:
                    choices_in_radius += seats[i][j]
                    total_seats += 1
        return choices_in_radius, total_seats

    seat_stats = list(map(_get_seats_stats, radii))
    choices_per_seat = list(map(lambda x: x[0] / x[1], seat_stats))
    return choices_per_seat


def seat_radii_prob(taken_data, data, max_radius=5, plot=False):
    """Plot seat radius information of preferred seat."""
    # We calculate the amount of choices per seat at each radius. We do so for
    # the data where some seats where unavailable, and where all seats were
    # available separately, then take the average.
    some_unavailable = get_seat_stats(
        form_answers.get_preferred_seats(taken_data), max_radius,
        taken=taken_seats.taken_seats)
    all_available = get_seat_stats(
        form_answers.get_preferred_seats(taken_data), max_radius)
    answers = np.array(some_unavailable) + np.array(all_available) / 2
    if plot:
        plt.plot(answers)
        plt.show()
    return answers


def beta_ratios(data):
    """Get the average β coefficients across all responses."""

    def ratio(form):
        all_fields = ["sitnexttofamiliar", "sitnexttoperson",
                      "sitgoodlocation", "siteasyreach"]
        all_fields = list(map(lambda key: form.get(key), all_fields))

        # Filter out invalid results.
        for field in all_fields:
            if field is None or (isinstance(field, str) and len(field) != 1):
                return None

        # Map each field to a fraction of all fields.
        all_fields = list(map(int, all_fields))
        return list(map(lambda field: field / sum(all_fields), all_fields))

    # [
    #   [β1/sum(β1..β4), β2/sum(β1..β4), β3/sum(β1..β4), β4/sum(β1..β4)],
    #     .  # Again, but for the second response.
    #     .
    #     .
    #   [...]
    # ]
    ratios = list(filter(lambda x: x is not None, map(ratio, data)))

    # The average of each column β_i for i=(1..4).
    avg_betas = np.mean(ratios, axis=0)
    # Return these four averages, scaled to sum to 1.
    return list(map(lambda x: x / sum(avg_betas), avg_betas))


if __name__ == "__main__":
    filenames = ["fri-form.json", "wed_24.json"]
    data = list(map(form_answers.load, filenames))

    answers = seat_radii_prob(data[0], data[1], plot=True)
    print("Seat choices at radius / seats at radius: {}".format(answers))

    all_data = list(sum(data, []))
    print("Average beta ratios: {}".format(
        json.dumps(beta_ratios(all_data), indent=4)))
