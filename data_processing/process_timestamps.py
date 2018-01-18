### still in progress

import datetime

output = open("output.txt", "w")
time = datetime.timedelta()

with open("timestamps_test.txt") as text:
    for line in text:
        number, time = line.split(",", 1)
        minutes, sec_millisec = time.split(" ")[1:3]
        sec, millisec = sec_millisec.split(",")

        print(datetime.timedelta(minutes = int(minutes), seconds = int(sec), milliseconds = int(millisec)))

        output.write(str(number) + " " + time)

        # update time
        new_time = datetime.timedelta(minutes = int(minutes), seconds = int(sec), milliseconds = int(millisec))
        time = time + new_time

output.close()
