from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import Student
import numpy as np

class TheaterModel(Model):
    '''
    Theater Seating Model
    '''

    def __init__(self, students = 288,
                 num_of_rows=12,
                 rows=[12, 12],
                 doors=[0],
                 sparsity=0.9):

        '''
        Create a new Theater Seating Model with given parameters.

        Args:
            students: Number of students in class
            num_of_rows: Number of rows of seating
            rows: Array listing the number of seats in each row for each section
                  split by the pathways
            doors: List of doors going clockwise from the bottom left (0, 1, 2, 3)
            sparsity: Weakness of friendship graph
        '''

        self.num_of_rows = num_of_rows
        self.rows = rows
        self.doors = doors

        # self.friendships = np.zeros([students, students], dtype=bool)

        self.theater = np.zeros([num_of_rows, sum(rows)], dtype=bool)
        self.paths = len(rows) - 1

        # Grid for visualization, width total number of seats in rows and paths,
        # height include space for doors
        # self.grid = MultiGrid(sum(rows) + self.paths, num_of_rows + 2, False)
        self.grid = MultiGrid(sum(rows), num_of_rows, False)

        self.schedule = RandomActivation(self)
        self.running = True

        # place agents at doorways initially
        for i in range(students):
            door = np.random.choice(self.doors)
            x = 0 if (door == 0) or (door == 1) else self.grid.width
            y = 0 if (door == 0) or (door == 3) else self.grid.height

            s = Student((x, y), self)
            self.schedule.add(s)
            self.grid.place_agent(s, (x, y))


    def step(self):
        ''' Advance the model one step '''

        # activate one (unseated) student at a time per step:
        agents = self.schedule.agents
        s = np.random.choice([a for a in agents if not a.found_seat])
        s.step()

        self.print_theater()

    def fill_graph(self, sparsity):
        ''' Fill out the friendship graph '''

        # Simple, random graph based on some sparsity
        s = self.friendships.shape[0]
        while(density < sparsity):
            x = np.random.randint(s)
            y = np.random.randint(s)
            self.friendships[x][y] = 1
            self.friendships[y][x] = 1
            density = np.sum(self.friendships) / (s**2)

    def find_accesible_seats(self):
        ''' Returns a list of seats avaliable without pushing past someone '''

        # flood fill to find accesible seats
        acc_seats = []
        p_ids = list(np.cumsum(self.rows))
        new_p_ids = [0] + p_ids

        # print('find_goal():')
        # print('Rows: {}, Num Paths: {}, p_ids = {}'.format(self.model.num_of_rows, self.model.paths, p_ids))
        for i in range(self.num_of_rows):
            row = self.theater[i, :]
            # print(row)
            # check right of paths:
            for j in range(self.paths):
                for x in np.arange(p_ids[j], p_ids[j+1]):
                    seat = row[x]
                    if seat == 0:
                        if (x, i) not in acc_seats:
                            acc_seats.append((x, i))
                    else:
                        # print('break1')
                        break

            # check left of paths:
            for j in range(self.paths):
                for x in np.arange(new_p_ids[j+1], new_p_ids[j], -1) - 1:
                    seat = row[x]
                    if seat == 0:
                        if (x, i) not in acc_seats:
                            acc_seats.append((x, i))
                    else:
                        break

        return acc_seats

    def print_theater(self):
        # t = model.theater
        p_ids = list(np.cumsum(self.rows))
        for i in np.arange(self.num_of_rows, 0, -1) - 1:
            line = '|'
            row_data = self.theater[i]
            for j, seat in enumerate(row_data):
                if j in p_ids:
                    line += ' '

                line += '_' if seat == 0 else 'o'

            if self.rows[-1] == 0:
                line += ' '

            print(line + '|')

        print('\n')
