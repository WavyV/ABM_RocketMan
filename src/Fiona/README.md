# The Classroom Model
Agent-Based Model of the seating process in a classroom

## Overview

The Classroom model consists of Seats having a particular attractiveness to each Student. Students are entering the classroom one by one every time step and choose the seat with the highest utility. This process continues until the total number of students in the class is reached or all seats are taken.

The utility function of seat i and student s is defined as the linear combination
``U(i,s) = b_p * p(i) + b_f * f(i,s) + b_s * s(i,s) + b_a * a(i)``
with coefficients b_p, b_f, b_s, b_a and the 4 utiltiy components
* p(i): positional utility
* f(i,s): friendship utility (depending on the neighboring students and their relationship to the decision-maker
* s(i,s): social utility (depending on the neighboring students and the sociability of the decision-maker)
* a(i): accessibility (depending on the number of students to be passed in order to reach the seat)


## The ClassroomModel class

Important attributes:

* classroom design as an instance of the ClassroomDesign class.
* coefficients of the utility function (default: [0.25, 0.25, 0.25, 0.25]
* probability distribution of the students' sociability, from which the sociability attribute is sampled when a Student is created (default: uniform)
* social network that defines friendship between students as binary connectivity matrix (default: random Erdos-Renyi-Network)
* seed for random number generation (default: 0)

The maxiumum number of students entering the classroom is set to the number of students in the social network. If no network is used it corresponds to the total number of seats in the classroom.

One model step consists in letting a new student enter the room and choose his favorit seat among all available seats.

## The ClassroomDesign class

The classroom is defined by

* list of blocks, where list entries specify the number of seats in a row per block. An aisle is created between all neighboring blocks. E.g. [0, 10, 10] represents two blocks with 10 seats in a row each, one aisle on the left, and one aisle between the two blocks.
* number of rows (note that this number includes horizontal aisles)
* horizontal aisles
* the position of entrances
* general desirability of each seat depending on its position in the room

## The Seat class 

Each seat object has the following characteristics which may change during a simulation

* the student currently occupying the seat
* position utility defined by the ClassroomDesign
* accessibility depending on the minimal number of students to be passed in order to reach the seat
* friendship utility based on friends in the neighborhood
* social utility based on the students sociability
* total utility depending on the model specific coefficients for the 4 components

## The Student class

* sociability: Preference for sitting next to people (1), aversion against sitting next to unknown people (-1), or being indifferent (0)
* initial happiness at the time the student made his seat choice. 'Happiness' is defined as the seat utility excluding the accessibility.
* will to change seat after already being seated (together with associated probability to move, and required threshold for increase in utility). This attribute enables repeated decision-making of already seated students and consequently adds more dynamics to the system.


## How to Run Simulations

First run the following to generate the data:

```
    $ python3 run_classroom_seating.py generate
```

And then you can visualize the seating process with:

```
    $ python3 run_classroom_seating.py animate
```

Some keys allow you to edit the iteration procedure:

p: pause/resume the animation
0: move to the start of the animation
e: move to the end of the animation
←: move 1 iteration left
→: move 1 iteration right
a: move 10 iteration left
d: move 10 iteration right

