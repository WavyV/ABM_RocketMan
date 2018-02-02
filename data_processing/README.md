# Data Processing

This folder contains Python code for analyzing and extracting useful data from
the data that was collected in classrooms. Here follows a short description of
each file:

`form_answers.py`: primary extraction of the collected data. Run this script
`python3 form_answers.py` and each form answer will be visualized.

`process_form.py`: this module relies on `form_answers.py` to process some of
the data into a more useful form, including generating distributions for social
and friendship attributes. Run with `python3 process_form.py` for
visualizations.

`process_timestamps.py`: is a script to convert timestamps we collected of
students entering classrooms into a Python friendly format.

`timestamps_test.txt`: is a test input for `process_timestamps.py`.

`output.txt`: is the corresponding output to `timestamps.txt`.
