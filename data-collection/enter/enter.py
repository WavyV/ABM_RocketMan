import json
import os
import time

"""
Usage:
    python3 enter.py

This will start a REPL. Enter a number and press enter to record it along with
the current time. Prefix an entered number with a ! to remove, this is in-case
of manual error. The data is saved automatically after every input. Ctrl-c to
quit.
"""


def repl(data_path):
    """Start the REPL."""
    data = dict()  # dict<int : unixtime>
    while True:
        user_input = input()
        # Delete entry.
        if user_input.startswith("!"):
            number = int(user_input[1:])
            del data[number]
            print("Deleted {}".format(number))
        # Insert entry.
        else:
            number = int(user_input)
            data[number] = time.time()
            print("Entered {}".format(number))
        # Always save.
        with open(data_path, "w") as f:
            json.dump(data, f, sort_keys=True, indent=4)
        print("Saved data")


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "events.json")
    repl(data_path)
