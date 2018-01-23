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
        sociability: will to sit next to unknown people (-1: unsociable, 0: indifferent, 1: sociable)
    """
    def __init__(self, unique_id, model, sociability=0):

        # set attributes
        super().__init__(unique_id, model)
        self.sociability = sociability

        # initial state of the student
        self.seated = False
        self.initial_happiness = 0

        """ currently not used """
        self.will_to_change_seat = False
        self.moving_threshold = 5 # TODO: think about appropriate value!
        self.moving_prob = 0.2 # TODO: think about appropriate value!



    """
    The seat selection procedure
    """
    def choose_seat(self, old_seat=None):

        # determine all possible seats to choose from
        seat_options = []
        for cell in self.model.grid.coord_iter():
            content, x, y = cell
            for agent in content:
                if type(agent) is Seat and agent.student is None:
                    seat_options.append(agent)


        if len(seat_options) == 0:
            print("No empty seats!")

        else:
            if self.model.random_seat_choice:
                # pick one randomly
                seat_choice = random.choice(seat_options)
            else:
                if old_seat is None:
                    # pick seat with highest utility (if multiple seats are optimal, choose one of them randomly)
                    seat_utilities = [seat.get_total_utility(self) for seat in seat_options]
                    seat_choice = random.choice(np.array(seat_options)[np.where(seat_utilities == np.max(seat_utilities))])

                    # move to the selected seat
                    seat_choice.student = self
                    self.model.grid.move_agent(self, seat_choice.pos)
                    self.initial_happiness = np.max(seat_utilities) - seat_choice.get_accessibility()
                    self.seated = True

                else:
                    """ only used if 'will_to_change_seat' is enabled """
                    # pick seat with highest utiltiy (if multiple seats are optimal, choose one of them randomly)
                    # include cost to get away from the old seat
                    seat_utilities = [seat.get_total_utility(self) - old_seat.get_stand_up_cost() for seat in seat_options]

                    if np.max(seat_utilities) - old_seat.get_total_utility(self) > self.moving_threshold:
                        # if the difference in utility between the current seat and another available seat exceeds the threshold, move to one of the optimal ones
                        seat_choice = random.choice(np.array(seat_options)[np.where(seat_utilities == np.max(seat_utilities))])

                        # move to the selected seat
                        self.model.grid.move_agent(self, seat_choice.pos)
                        self.initial_happiness = np.max(seat_utilities) - seat_choice.get_accessibility()
                        seat_choice.student = self

                        # make old seat available again
                        agent.occupied = False



    """
    At each tick the student either selects a seat or stays at its current seat
    """
    def step(self):

        # only choose another seat if not seated yet or
        # if the student has the characteristic to change its seat again
        if not self.seated:
            # choose and move to seat
            self.choose_seat()

        if self.will_to_change_seat:
            # with the given probability the student searches for a better seat
            r = random.random()
            if r < self.moving_prob:
                # compare current seat utility with all other available seats. If there is a much better one, move
                content = self.model.grid.get_cell_list_contents(self.pos)
                for agent in content:
                    if type(agent) is Seat:
                        self.choose_seat(agent)


class Seat(Agent):

    """
    Create a seat

    Args:
        model: the associated ClassroomModel
    """
    def __init__(self, model):

        self.model = model
        self.student = None

    """
    Get the position utility of the seat, based on its location in the Classroom
    """
    def get_position_utility(self):
        return self.model.classroom.pos_utilities[self.pos]

    """
    Get the accessibility of the seat defined as the negative number of Students between the seat and the aisle.
    The student count for the left and right side are compared and the minimum is used.

    Returns:
        u_accessibility: utiltiy (inverse of the number of Students to pass in order to reach the seat)
    """
    def get_accessibility(self):

        x, y = self.pos

        dist_to_aisles = np.array([(x - i) for i in self.model.classroom.aisles_x])

        # determine the number of people sitting between this seat and the closest aisle on the left side
        dist_to_aisles_left = dist_to_aisles[np.where(dist_to_aisles > 0)]
        if len(dist_to_aisles_left) > 0:
            aisles_left = np.array(self.model.classroom.aisles_x)[np.where(dist_to_aisles > 0)]
            aisle_left = aisles_left[np.argmin(dist_to_aisles_left)]
            row_left = [(i, y) for i in range(aisle_left + 1, x)]

            count_left = 0
            for agent in self.model.grid.get_cell_list_contents(row_left):
                if type(agent) is Student:
                    count_left += 1

        else:
            # if there is no aisle on the left side, always count the student on the right side
            count_left = np.infty

        # determine the number of people sitting between this seat and the closest aisle on the right side
        dist_to_aisles_right = dist_to_aisles[np.where(dist_to_aisles < 0)]
        if len(dist_to_aisles_right) > 0:
            aisles_right = np.array(self.model.classroom.aisles_x)[np.where(dist_to_aisles < 0)]
            aisle_right = aisles_right[np.argmin(abs(dist_to_aisles_right))]
            row_right = [(i, y) for i in range(x + 1, aisle_right)]

            count_right = 0
            for agent in self.model.grid.get_cell_list_contents(row_right):
                if type(agent) is Student:
                    count_right += 1

        else:
            # if there is no aisle on the right side, always count the student on the left side
            count_right = np.infty

        u_accessibility = -min(count_right, count_left)

        return u_accessibility


    """
    Get the local neighborhood around the Seat

    Args:
        size_x: width of the neighborhood
        size_y: height of the neighborhood

    Returns:
        neighborhood: matrix containing IDs of neighboring students (-1 means empty or no seat at all)
    """
    def get_neighborhood(self, size_x, size_y):

        x, y = self.pos

        # assure that size_x and size_y are uneven, so that the neighborhood has one central seat
        if size_x%2 == 0:
            size_x -= 1
        if size_y%2 == 0:
            size_y -= 1

        neighborhood = -np.ones((size_x, size_y))
        to_center_x = int(size_x/2)
        to_center_y = int(size_y/2)

        for i in range(size_x):
            for j in range(size_y):

                coords = (x + i - to_center_x, y + j - to_center_y)

                # Skip if new coords out of bounds.
                if self.model.grid.out_of_bounds(coords):
                    continue
                else:
                    for agent in self.model.grid.get_cell_list_contents(coords):
                        if type(agent) is Student and agent.seated:
                            neighborhood[i,j] = agent.unique_id

        return neighborhood

    """
    Get the social utility of the Seat (including friendship and sociability component).
    Higher values represent higher attractivity.
    Based on the social network connections to neighboring Students,
    and the individual sociability of the Student making the decision:
        - The more friends are sitting closeby, the higher the friendship utility.
        - If the Student is sociable (value = 1), the presence of unfamiliar neighbors increases the utility
        - If the Student is unsociable (value = -1), the presence of unfamiliar neighbors decreases the utility

    Args:
        student: the student making the seating choice

    Returns:
        u_friendship: friendship component of the social utility
        u_sociability: sociability component of the social utility
    """
    def get_social_utility(self, student):

        interaction_x, interaction_y = self.model.friendship_interaction_matrix.shape

        neighborhood = self.get_neighborhood(interaction_x, interaction_y)

        u_friendship = 0
        u_sociability = 0

        for x in range(interaction_x):
            for y in range(interaction_y):

                if neighborhood[x,y] >= 0: # if there is a student

                    # compute the friendship component
                    friendship = self.model.social_network[int(student.unique_id), int(neighborhood[x,y])]
                    u_friendship += self.model.friendship_interaction_matrix[x,y] * friendship

                    # if neighbouring seat is occupied by a student that is not a friend,
                    # determine the sociability component based on the student's sociability attribute
                    if friendship == 0:
                        u_sociability += self.model.sociability_interaction_matrix[x,y] * student.sociability

        return u_friendship, u_sociability

    """
    Get the overall utility of the Seat as a linear combination of position, friendship, sociability and accessibility components

    Returns:
        total_utility: high values represent high attractivity of the seat
    """
    def get_total_utility(self, student):

        friendship_component, sociability_component = self.get_social_utility(student)
        coef_p, coef_f, coef_s, coef_a = self.model.coefs
        total_utility = coef_p * self.get_position_utility() + coef_f * friendship_component + coef_s * sociability_component + coef_a * self.get_accessibility()

        return total_utility



class ClassroomModel(Model):

    """
    Create a classroom model

    Args:
        N: maximal number of students entering the classroom
        classroom_design: instance of ClassroomDesign defining the layout and position-dependent seat utilities of the classroom
        coefs: list [coef_p, coef_f, coef_s, coef_a] defining the coefficients for the position, friendship, sociability and accessibility components in the utility function
        social_network: the underlying social network as a connectivity matrix (1 means friendship, 0 means indifference). If not given, all connections are set to zero.
        seed: seed for the random number generation
    """
    def __init__(self, N, classroom_design, coefs, social_network=None, seed=0):

        self.max_num_agents = N
        self.classroom = classroom_design
        self.coefs = coefs

        # if all utility components are set to zero, seat choices are completely random.
        self.random_seat_choice = np.all(coefs == 0)

        if social_network is None and coefs[1] == 0:
            # no social connections
            self.social_network = np.zeros((N,N))
        elif social_network is None and coefs[1] != 0:
            # create a random network
            self.social_network = np.random.randint(2, size=(N,N))
        else:
            self.social_network = social_network

        random.seed(seed)

        self.grid = MultiGrid(classroom_design.width, classroom_design.num_rows, False)
        self.schedule = RandomActivation(self)

        # matrices that determine the importance of neighboring seats for the social utility
        # They need to have the same shape
        self.friendship_interaction_matrix = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]).T
        self.sociability_interaction_matrix = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]).T

        # initialize seats (leave aisles free)
        for x in range(self.classroom.width):
            if not x in self.classroom.aisles_x:
                for y in range(self.classroom.num_rows):
                    if not y in self.classroom.aisles_y:
                        # create new seat
                        seat = Seat(self)
                        self.grid.place_agent(seat, (x,y))


    """
    Advance the model by one step. If the maximum student number is not reached yet, create a new student every tick.
    """
    def step(self):

        # as long as the max number of students is not reached, add a new one
        n = self.schedule.get_agent_count()
        if n < self.max_num_agents:
            # create student
            if self.coefs[2] != 0:
                sociability = random.choice([-1,0,1])
                student = Student(n, self, sociability)
            else:
                student = Student(n, self)

            self.schedule.add(student)

            # place new student randomly at one of the entrances
            initial_pos = random.choice(self.classroom.entrances)
            self.grid.place_agent(student, initial_pos)

        self.schedule.step()




class ClassroomDesign():

    """
    Create a classroom layout composed of aisles and entrances

    Args:
        blocks: list [b_1,b_2,...,b_n] defining the number of seats per row in each block b_i. Between two blocks there is an aisle.
        num_rows: number of rows in the classroom. Includes horizontal aisles.
        pos_utilities: matrix that assigns a positional utility value to each seat in the classroom
        entrances: list of tuples representing the positions of entrances in the classroom
        aisles_y: list [y_1,y_2,...,y_m] defining the rows where horizontal aisles are
    """
    def __init__(self, blocks, num_rows, pos_utilities=None, entrances=None, aisles_y=None):

        self.width = sum(blocks) + len(blocks) - 1
        self.num_rows = num_rows

        # define vertical aisles
        self.aisles_x = []
        current_x = 0
        for b in range(len(blocks)-1):
            current_x += blocks[b]
            self.aisles_x.append(current_x)
            current_x += 1

        # define horizontal aisle in the front
        if aisles_y is None:
            self.aisles_y = [0]
        else:
            self.aisles_y = aisles_y

        # define entrances
        if entrances is None:
            self.entrances = [((0, 0)),((width-1,0))]
        else:
            self.entrances = entrances

        # define utility/attractivity of each seat location
        if pos_utilities is not None and pos_utilities.shape == (self.width, num_rows):
            self.pos_utilities = pos_utilities
        else:
            self.pos_utilities = np.zeros((self.width, num_rows))
