#!/usr/bin/python3

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
import numpy as np


class Student(Agent):

    """
    Create a student with individual characteristics

    Args:
        unique_id: student identifier
        model: the associated ClassroomModel
    """
    def __init__(self, unique_id, model, use_sociability):

        super().__init__(unique_id, model)

        self.seated = False
        self.will_to_change_seat = False
        self.random_seat_choice = False

        if use_sociability:
            self.sociability = random.choice([-1,0,1])
        else:
            self.sociability = 0

    """
    The seat selection procedure
    """
    def choose_seat(self):

        # determine all possible seats to choose from
        seat_options = []

        for cell in self.model.grid.coord_iter():
            content, x, y = cell
            for agent in content:
                if type(agent) is Seat and not agent.occupied:
                    seat_options.append(agent)


        if len(seat_options) == 0:
            print("No empty seats!")
            return
        else:
            if self.random_seat_choice:
                # pick one randomly
                seat_choice = random.choice(seat_options)
            else:
                # pick seat with highest preference (if multiple seats are optimal, choose one of them randomly)
                seat_utilities = [seat.get_total_utility(self) for seat in seat_options]
                seat_choice = random.choice(np.array(seat_options)[np.where(seat_utilities == np.max(seat_utilities))])

            # move to the selected seat
            seat_choice.occupied = True
            self.model.grid.move_agent(self, seat_choice.pos)
            self.seated = True




    """
    At each tick the student either selects a seat or stays at its current seat
    """
    def step(self):

        # only choose another seat if not seated yet or
        # if the student has the characteristic to change its seat again
        if not self.seated or self.will_to_change_seat:

            # TODO: if will_to_change_seat: include cost to get away from the current seat

            # make old seat available again
            content = self.model.grid.get_cell_list_contents(self.pos)
            for agent in content:
                if type(agent) is Seat:
                    agent.occupied = False

            # choose and move to new seat
            self.choose_seat()


class Seat(Agent):

    """
    Create a seat

    Args:
        model: the associated ClassroomModel
    """
    def __init__(self, model):

        self.model = model
        self.occupied = False

    """
    Get the position utility of the seat, based on its location in the Classroom
    """
    def get_position_utility(self):
        return self.model.classroom.seats[self.pos]

    """
    Get the number of Students that have to stand up in order to reach this Seat
    Determined relative to the closest aisle

    Returns:
        count: number of Students
    """
    def get_stand_up_cost(self):

        x, y = self.pos

        closest_aisle = np.argmin([abs(x - i) for i in self.model.classroom.aisles_x])
        aisle_x = self.model.classroom.aisles_x[closest_aisle]
        if aisle_x > x:
            row = [(i, y) for i in range(x + 1, aisle_x)]
        else:
            row = [(i, y) for i in range(aisle_x + 1, x)]

        count = 0
        for agent in self.model.grid.get_cell_list_contents(row):
            if type(agent) is Student:
                count += 1

        return count


    """
    Get the local neighborhood around the Seat

    Returns:
        neighborhood: matrix containing IDs of neighboring students (-1 means empty or no seat at all)
    """
    def get_neighborhood(self, radius):

        x, y = self.pos
        neighborhood = -np.ones((2*radius + 1, 2*radius + 1))

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):

                coords = (dx + x, dy + y)

                # Skip if new coords out of bounds.
                if self.model.grid.out_of_bounds(coords):
                    continue
                else:
                    for agent in self.model.grid.get_cell_list_contents(coords):
                        if type(agent) is Student and agent.seated:
                            neighborhood[dx + radius, dy + radius] = agent.unique_id

        return neighborhood

    """
    Get the social utility of the Seat.
    Based on the social network connections to neighboring Students,
    and the sociability of the Student making the decision:
        - The more friends are sitting closeby, the higher the seat utility. The influence fades with distance.
        - If the Student is sociable (value = 1), the presence of direct neighbors increases the utility
        - If the Student is unsociable (value = -1), the presence of direct neighbors decreases the utility

    Returns:
        utility: high values represent attractivity
    """
    def get_social_utility(self, student):

        interaction_radius = int(self.model.interaction_matrix.shape[0]/2)
        neighborhood = self.get_neighborhood(interaction_radius)

        utility = 0

        for k in range(-interaction_radius, interaction_radius + 1):
            for l in range(-interaction_radius, interaction_radius + 1):

                x = k + interaction_radius
                y = l + interaction_radius

                if neighborhood[x,y] >= 0: # if there is a student
                    friendship = self.model.social_network[int(student.unique_id), int(neighborhood[x,y])]
                    utility += self.model.interaction_matrix[x,y] * friendship

                    # if direct neighbor and no social connection, adjust preference based on the student's sociability
                    if abs(k) == 1 and l == 0 and friendship == 0:
                        utility += student.sociability

        return utility

    """
    Get the overall utility of the Seat as a combination of position, social utility and "stand-up-cost"

    Returns:
        total_utility: high values represent attractivity
    """
    def get_total_utility(self, student):

        total_utility = self.get_position_utility() + self.get_social_utility(student) - self.get_stand_up_cost()
        return total_utility



class ClassroomModel(Model):

    """
    Create a classroom model

    Args:
        N: maximal number of students entering the classroom
        height, width: dimensions of the classroom
        social_network: the underlying social network as a connectivity matrix (1 means friendship, 0 means indifference)
    """
    def __init__(self, N, height, width, seed, use_sociability, social_network=None):
        self.max_num_agents = N
        self.grid = MultiGrid(width, height, False)
        self.classroom = ClassroomDesign(width, height)
        random.seed(seed)
        self.use_sociability = use_sociability
        self.schedule = RandomActivation(self)

        self.interaction_matrix = np.array([[0.25, 0.5, 0.25], [1, 0, 1], [0.25, 0.5, 0.25]]).T

        if social_network is None:
            self.social_network = np.zeros((N,N))
        else:
            self.social_network = social_network

        # initialize seats (leave aisles free)
        for x in range(width):
            if not x in self.classroom.aisles_x:
                for y in range(height):
                    if not y in self.classroom.aisles_y:
                        # create new seat with a position specific utility
                        seat = Seat(self)
                        self.grid.place_agent(seat, (x,y))


    """
    Advance the model by one step. If the maximum student number is not reached yet, create a new student every tick.
    """
    def step(self):

        # as long as the max number of students is not reached, add a new one
        n = self.schedule.get_agent_count()
        if n < self.max_num_agents:
            student = Student(n, self, self.use_sociability)
            self.schedule.add(student)

            # place new student randomly at one of the entrances
            initial_pos = random.choice(self.classroom.entrances)
            self.grid.place_agent(student, initial_pos)

        self.schedule.step()


class ClassroomDesign():

    """
    Create a classroom layout composed of aisles and entrances

    Args:
        width, height: dimensions of the classroom
    """
    def __init__(self, width, height):

        # define vertical aisles
        self.aisles_x = [int(width/3),2*int(width/3)]

        # define horizontal aisles
        self.aisles_y = [0]

        # define entrances
        self.entrances = [((0, 0)),((width-1,0))]

        # define utility/attractivity of each seat location (TODO: assign sensible values)
        self.seats = np.zeros((width, height))
