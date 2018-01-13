#!/usr/bin/python3

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
import numpy as np


class Student(Agent):

    """ A student with its characteristics """
    def __init__(self, unique_id, model):

        super().__init__(unique_id, model)

        self.seated = False
        self.will_to_change_seat = True

    """ the selection procedure """
    def choose_seat(self):

        # determine all possible seats to choose from
        seat_options = []

        for cell in self.model.grid.coord_iter():
            content, x, y = cell
            for agent in content:
                if type(agent) is Seat and not agent.occupied:
                    seat_options.append(agent)

        # pick one randomly (TODO: integrate preferences)
        if len(seat_options) > 0:
            seat_choice = random.choice(seat_options)
            seat_choice.occupied = True
            self.model.grid.move_agent(self, seat_choice.pos)
            self.seated = True
        else:
            print("No empty seats!")

    """ at each tick the student picks a seat (randomly or based on prefernce) """
    def step(self):

        if not self.seated or self.will_to_change_seat:

            # make old seat available again
            content = self.model.grid.get_cell_list_contents(self.pos)
            for agent in content:
                if type(agent) is Seat:
                    agent.occupied = False

            # choose and move to new seat
            self.choose_seat()


class Seat(Agent):

    """ a seat with its characteristics """
    def __init__(self, model, utility):

        self.model = model
        self.utility = utility
        self.num_neighbors = 0
        self.occupied = False

    """ determine the number of occupied seats in the neighborhood """
    def get_neigbors(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        self.num_neighbors = sum([(1 if type(i) is Student else 0) for i in neighborhood])



class ClassroomModel(Model):

    """A model with a maximal number of students entering the classroom """
    def __init__(self, N, width, height):
        self.max_num_agents = N
        self.grid = MultiGrid(width, height, False)
        self.classroom = ClassroomDesign(width, height)
        self.schedule = RandomActivation(self)

        # initialize seats (leave aisles free)
        for x in range(width):
            if not x in self.classroom.aisles_x:
                for y in range(height):
                    if not y in self.classroom.aisles_y:
                        # create new seat with a position specific utility
                        seat = Seat(self, self.classroom.seats[x,y])
                        self.grid.place_agent(seat, (x,y))


    '''Advance the model by one step.'''
    def step(self):

        # as long as the max number of students is not reached, add a new one
        n = self.schedule.get_agent_count()
        if n < self.max_num_agents:
            student = Student(n, self)
            self.schedule.add(student)

            # place new student randomly at one of the entrances
            initial_pos = random.choice(self.classroom.entrances)
            self.grid.place_agent(student, initial_pos)

        self.schedule.step()


class ClassroomDesign():

    """ a classroom layout composed of aisles and entrances """
    def __init__(self, width, height):

        # define vertical aisles
        self.aisles_x = [int(width/2)]

        # define horizontal aisles
        self.aisles_y = [0]

        # define entrances
        self.entrances = [((0,0)),((width-1,0))]

        # define utility/attractivity of each seat location (TODO: assign sensible values)
        self.seats = np.zeros((width, height))
