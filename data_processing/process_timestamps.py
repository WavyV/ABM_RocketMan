import datetime
import json


def convert_timestamps(in_filename):
    """Convert timestamps from in_filename and return them."""
    all_timedeltas = []
    time = datetime.timedelta(0)
    with open(in_filename) as text:
        for line in text:
            number, min_sec = line.split(",", 1)
            minutes, sec_millisec = min_sec.split(" ")[1:3]
            sec, millisec = sec_millisec.split(",")

            all_timedeltas.append(time)

            # update time
            new_time = datetime.timedelta(minutes=int(minutes),
                                          seconds=int(sec),
                                          milliseconds=int(millisec))
            # print(newtime)
            time = time + new_time
    return all_timedeltas


if __name__ == "__main__":
    all_timedeltas = convert_timestamps("timestamps_test.txt")
    str_output = list(map(lambda i: "{} {}".format(i[0], i[1]),
                          enumerate(all_timedeltas)))
    with open("output_timedeltas.txt", "w") as f:
        json.dump(str_output, f)
