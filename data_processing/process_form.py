import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import form_answers
import taken_seats

"""
Process form answers into useful information for the model.

Form answers are calculated using the form_answers module.
"""


def scale(old_val, old_min, old_max, new_min, new_max):
    """Scale the given x from the old range to the new range."""
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((old_val - old_min) * new_range) / old_range) + new_min


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
        plt.title("#choices at radius / #seats at radius")
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


def seat_location_scores(convolution_iters=10, plot=False):
    """Return a 2d array (row major) of location scores for each seat.

    Applies convolution to the "preferred seat" question data.
    """
    seats = form_answers.get_preferred_seats(form_answers.ALL_DATA)
    weights = [
        [0.25, 0.5, 0.25],
        [0.5,  1,   0.5],
        [0.25, 0.5, 0.25],
    ]
    for _ in range(convolution_iters):
        seats = signal.convolve2d(seats, weights, boundary="symm", mode="same")
    # Scale each seat into [0 1].
    max_score = np.amax(seats)
    min_score = np.amin(seats)
    for i in range(len(seats)):
        for j in range(len(seats[0])):
            seats[i][j] = scale(seats[i][j], min_score, max_score, 0, 1)
    if plot:
        plt.imshow(seats)
        plt.colorbar()
        plt.show()
    return seats


def agent_attribute_gen(hist_data, scale_to=None, noise_std_dev=0.25):
    """Return a generator that yields values based on given histogram data.

    `noise_std_dev` is the std deviations of noise measured in bin sizes.
    """
    bin_heights, bin_ranges, _ = hist_data
    bin_probs = [x/sum(bin_heights) for x in bin_heights]
    bin_indices = list(range(len(bin_heights)))
    bin_max = max(bin_ranges)
    bin_min = min(bin_ranges)
    bin_width = abs(bin_ranges[0] - bin_ranges[1])
    while True:
        # Pull a value from the histogram.
        bin_index = np.random.choice(bin_indices, p=bin_probs)
        lower_bin_range = bin_ranges[bin_index]
        upper_bin_range = bin_ranges[bin_index + 1]
        drawn_value = np.random.uniform(lower_bin_range, upper_bin_range)

        # Generate some gaussian noise.
        noise_scale = noise_std_dev * bin_width
        noise = np.random.normal(loc=0, scale=noise_scale)

        # We ignore the value if it's out of bounds.
        old_value = drawn_value + noise
        if old_value < bin_min or old_value > bin_max:
            continue

        # Scale the noisy value within the given scale.
        if scale is not None:
            min_val, max_val = scale_to
            yield scale(old_value, bin_min, bin_max, min_val, max_val)
        else:
            yield old_value


def agent_sociability_gen():
    """Return a generator that yields agent sociability attributes.
    These yielded values are scaled within [0 1].

    Usage:
        sociability_gen = agent_sociability_gen()
        first_sociability = next(sociability_gen)
        second_sociability = next(sociability_gen)
        etc..

    Sociability is calculated by sampling from the histogram bins, the
    probability of choosing a bin is proportional to its size.
    """
    _, hist_data = form_answers.importance_of_person(form_answers.ALL_DATA)
    return agent_attribute_gen(hist_data, scale_to=(0, 1))


def agent_friends_gen():
    """Return a generator that yields agent course friends.

    Usage:
        friends_gen = agent_friends_gen()
        first_friends = next(friends_gen)
        second_friends = next(friends_gen)
        etc..

    Friends are calculated by sampling from the histogram bins, the probability
    of choosing a bin is proportional to its size. Then a value is chosen
    uniformly from that bin's range.
    """
    _, hist_data = form_answers.course_friends(form_answers.ALL_DATA)
    return agent_attribute_gen(hist_data, scale_to=(0, 50))


if __name__ == "__main__":
    print("Average beta ratios: {}".format(
        json.dumps(beta_ratios(form_answers.ALL_DATA), indent=4)))

    answers = seat_radii_prob(
        form_answers.DATA[0], form_answers.DATA[1], plot=True)
    print("#choices at radius / #seats at radius: {}".format(answers))

    sociability_gen = agent_sociability_gen()
    plt.hist([next(sociability_gen) for _ in range(1000)], bins=100)
    plt.title("Generated sociability")
    plt.show()

    friendship_gen = agent_friends_gen()
    plt.hist([next(friendship_gen) for _ in range(75)], bins=100)
    plt.title("Generated course friends")
    plt.show()

    seat_location_scores(plot=True)
