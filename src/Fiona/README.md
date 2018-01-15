# The Classroom Model
Agent-Based Model of the seating process in a classroom

## Overview

The Classroom model consists of Seats having a particular attractiveness to each Student. Students are entering the classroom one by one every time step and choose the seat with the highest preference. This process continues until the predefined maximal number of students is reached or the all seats are occupied.

## Important Model Attributes

* maximal number of students
* number of rows
* classroom layout as a list of blocks. List entries specify the number of columns per block. An aisle is created between all neighboring blocks. E.g. [0, 10, 10] represents two blocks with 10 seats in a row each, one aisle on the left, and one aisle between the two blocks.
* seed for random number generation
* social network that describes friendship between students
* interaction matrix determining how much influence which position in the neighborhood has on the friendship component of the social utility

## Important Seat Attributes

* position utility based on the location in the classroom
* stand-up-cost: minimal number of students to pass in order to reach the seat
* social utility based on friends in the neighborhood and the students sociability
* total utility

## Important Student Attributes

* will to change seat after already being seated (together with associated probability to move, and required threshold for increase in utility)
* sociability: Preference for sitting next to people, avoid sitting next to unknown people, or being indifferent

## Simulations

Different simulations can be performed:
* completely random seat choices: ``random_seat_choice=True``
* positional preferences only (location in classroom and blocking): ``random_seat_choice=False``, ``use_sociability=False``, and do not define a social network
* preferences based on position and individual sociability: ``random_seat_choice=False``, ``use_sociability=True``, and do not define a social network
* preferences based on position, individual sociability, and friendship: ``random_seat_choice=False``, ``use_sociability=True``, and define a social network

## How to Run Simulations

To run the model and visualize the seating process, run

```
    $ python3 run_classroom_seating.py
```

### To be determined

* distribution of sociability
* weighting of positional and social components
* interaction matrix?
* social network
