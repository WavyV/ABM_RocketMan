# The Classroom Model
In this project an Agent-Based Model (ABM) of the seating process in classrooms is implemented.

## Overleaf LaTeX Report Link: 
https://www.overleaf.com/13250817fxtnzrkyyrjf

## Structure

* ``model.py`` implements the classroom model.
* ``run_model.py`` provides methods to run model simulations. Default model inputs (position utilities, sociabilities, friendship) can be loaded from ``model_input``.
* ``animation.py`` visualizes the seating process of a simulation (loading the simulation data from ``animation_data``).
* ``model_comparison.py`` provides methods to analyse and compare model outputs (seating patterns).
* ``parameter_estimation.py`` implements the estimation of utility coefficients using the SPSA algorithm together with the collected data. Results (used coefficients and the respected errors) are saved to ``model_output/parameter_estimation``
* `sensitivity-analysis.py` used to analyze the model using OFAT and Sobol techniques, and visualize the analysis.
* ``social`` provides methods to generate realistic social networks.
* ``data-collection`` contains the implementation of the online survey.
* ``data`` contains the results from the online survey conducted during UvA lectures.
* ``data_processing`` provides methods to process the data collected during UvA lectures.

## A Few Quick Commands

NOTE: before running any commands: `pip3 install -r requirements.txt`.

`python3 animation.py` to run the interactive model, see interaction commands below.

`python3 data_processing/form_answers.py` to see visualized answers from data collection.

`python3 data_processing/process_form.py` for further visualized information based on the form answers.

`python3 sensitivity_analysis.py --ofat-analysis` to run the OFAT SA visualizations.

`python3 sensitivity_analysis.py --sobol-analysis` to run the Sobol SA visualizations.

## Model Overview

The Classroom model consists of Seats having a particular attractiveness to each Student. Students are entering the classroom one by one every time step and choose the seat with the highest utility. This process continues until the total number of students in the class is reached or all seats are taken.

The utility function of seat i and student s is defined as the linear combination

``U(i,s) = b_p * p(i) + b_f * f(i,s) + b_s * s(i,s) + b_a * a(i)``

with coefficients b_p, b_f, b_s, b_a and the 4 utiltiy components
* p(i): positional utility
* f(i,s): friendship utility (depending on the neighboring students and their relationship to the decision-maker
* s(i,s): social utility (depending on the neighboring students and the sociability of the decision-maker)
* a(i): accessibility (depending on the number of students to be passed in order to reach the seat)


### The ClassroomModel class

Important attributes:

* classroom design as an instance of the ClassroomDesign class.
* coefficients of the utility function (default: [0.25, 0.25, 0.25, 0.25])
* a sequence of sociability values from which the sociability attribute of a new student is determined. (Default: sequence with values drawn from a uniform distribution of range [0,1])
* degree sequence (number of friends per student) based on which the social network is generated. (Default: random Erdos-Renyi-Network)
* seed for random number generation. (Default: 0)

The maxiumum number of students entering the classroom is set to the number of students in the social network. If no network is used it corresponds to the total number of seats in the classroom.

One model step consists of letting a new student enter the room and choose his favorit seat among all available seats.

### The ClassroomDesign class

The classroom is defined by

* list of blocks, where list entries specify the number of seats in a row per block. An aisle is created between all neighboring blocks. E.g. [0, 10, 10] represents two blocks with 10 seats in a row each, one aisle on the left, and one aisle between the two blocks.
* number of rows (note that this number includes horizontal aisles)
* horizontal aisles
* the position of entrances
* general desirability of each seat depending on its position in the room

### The Seat class 

Each seat object has the following characteristics which may change during a simulation

* the student currently occupying the seat
* position utility defined by the ClassroomDesign
* accessibility depending on the minimal number of students to be passed in order to reach the seat
* friendship utility based on friends in the neighborhood
* social utility based on the students sociability
* total utility depending on the model specific coefficients for the 4 components

### The Student class

* sociability: Preference for sitting next to people. A sociability value of zero always represents indifference. Aversion can be modeled by allowing negative values.
* initial happiness at the time the student made his seat choice. 'Happiness' is defined as the seat utility excluding the accessibility.
* will to change seat after already being seated (together with associated probability to move, and required threshold for increase in utility). This attribute enables repeated decision-making of already seated students and consequently adds more dynamics to the system. In the model we study during this project, this property is not used.


### How to Run Simulations

#### Advance the Model Step by Step and Analyse or Process the Model States

In general, use the ClassroomModel() constructor to create a model with the desired properties. Then the model can be advanced using ``model.step()``. At any step the current seating pattern (``model.get_binary_model_state()``) can be analysed with the tools in ``model_comparison.py``. The complete classroom state (including aisles, utilities, etc.) can be obtained by the ``get_model_state(model)`` method in ``run_model.py``.

#### Animation of the Simulated Seating Process

First run the following to generate some simulation data.

```
    $ python3 run_model.py [file_name]
```
The model states at each time point will be saved to ``animation_data/model_data.json`` or if specified to ``animation_data/file_name``.
To simulate a customized model instead of the default one, create a ClassroomModel instance with the desired characteristics. Then run the ``generate_data()`` method to generate the data for animation.

And then you can visualize the seating process of the generated data with:

```
    $ python3 animation.py [file_name]
```

During the animation some keys allow you to edit the iteration procedure:

* p: pause/resume the animation
* 0: move to the start of the animation
* e: move to the end of the animation
* ←: move 1 iteration left
* →: move 1 iteration right
* a: move 10 iteration left
* d: move 10 iteration right

