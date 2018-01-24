import json
import sys

import matplotlib.pyplot as plt

import form_answers
import taken_seats


def get_seat_stats(seats, taken, center=(3, 12), max_radius=5,
                   exclude_taken=True, plot=False):
    """Return a tuple(total_choices, prob_seat).

    NOTE: Arguments seats and taken should exclude aisles!

    total_choices is the total amount of seat choices at each radius.
    prob_seat is this value divided by the amount of seats at that radius.
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
                exclude_seat = exclude_taken and taken[i][j]
                if not exclude_seat:
                    choices_in_radius += seats[i][j]
                    total_seats += 1
        # Consider seats d_i away from center.
        # We need to avoid the corners already counted.
        for i in set([c_i + radius, c_i - radius]):
            for d_j in range(-(radius - 1), radius):
                j = c_j + d_j
                if not exclude_seat:
                    choices_in_radius += seats[i][j]
                    total_seats += 1
        return choices_in_radius, total_seats

    seat_stats = list(map(_get_seats_stats, radii))
    total_choices_at_radius = list(map(lambda x: x[0], seat_stats))
    avg_choices_per_seat = list(map(lambda x: x[0] / x[1], seat_stats))
    if plot:
        plt.plot(radii, total_choices_at_radius)
        plt.plot(radii, avg_choices_per_seat)
        plt.show()
    return total_choices_at_radius, avg_choices_per_seat


def plot_seat_radii(data):
    """Plot seat locations and radius stats."""
    seats = form_answers.get_preferred_seats__some_unavailable(data)
    taken = taken_seats.taken_seats
    seat_stats = get_seat_stats(seats, taken, exclude_taken=True, plot=True)
    print("Seat choices at radius: {}".format(seat_stats[0]))
    print("Seat choices at radius / seats at radius: {}".format(seat_stats[1]))


def beta_ratios(data):
    """Get the average ratio of the β coefficients."""

    def ratio(form):
        all_fields = ["sitnexttofamiliar", "sitnexttoperson", "sitgoodlocation", "siteasyreach"]
        all_fields = list(map(lambda key: form.get(key), all_fields))

        # Filter out bad results.
        for field in all_fields:
            if field is None or len(field) != 1:
                return None

        # Map to int and return each field as a fraction of total.
        all_fields = list(map(int, all_fields))
        return list(map(lambda field: field / sum(all_fields), all_fields))

    return list(map(ratio, data))


if __name__ == "__main__":
    data = form_answers.load(sys.argv[1])
    plot_seat_radii(data)
    print(json.dumps(beta_ratios(data), indent=4))
