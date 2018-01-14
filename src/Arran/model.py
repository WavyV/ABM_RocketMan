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

    seated_students = []

    def __init__(self, students = 100,
                 num_of_rows=12,
                 blocks=[8, 8, 8],
                 doors=[0],
                 sparsity=0.1):

        '''
        Create a new Theater Seating Model with given parameters.

        Args:
            students: Number of students in class
            num_of_rows: Number of rows of seating
            blocks: Array listing the number of seats in each row for each section
                    split by the pathways
            doors: List of doors going clockwise from the bottom left (0, 1, 2, 3)
            sparsity: Weakness of friendship graph
        '''

        self.num_of_rows = num_of_rows
        self.blocks = blocks
        self.doors = doors

        self.friendships = np.zeros([students, students], dtype=bool)
        self.fill_graph(sparsity)

        self.theater = np.zeros([num_of_rows, sum(blocks)], dtype=bool)
        self.paths = len(blocks) - 1

        # Grid for visualization, width total number of seats in rows and paths,
        # height include space for doors
        # self.grid = MultiGrid(sum(rows) + self.paths, num_of_rows + 2, False)
        self.grid = MultiGrid(sum(blocks), num_of_rows, False)
        self.plan = np.zeros([num_of_rows, sum(blocks) + self.paths])

        for b in (np.cumsum(blocks) + np.arange(len(blocks)))[:-1]:
            self.plan[:, b] = -1

        self.schedule = RandomActivation(self)
        self.running = True

        # place agents at doorways initially
        for i in range(students):
            door = np.random.choice(self.doors)
            x = 0 if (door == 0) or (door == 1) else self.grid.width
            y = 0 if (door == 0) or (door == 3) else self.grid.height

            s = Student((x, y), i, self)
            self.schedule.add(s)
            self.grid.place_agent(s, (x, y))


    def step(self):
        ''' Advance the model one step '''

        # activate one (unseated) student at a time per step:
        agents = self.schedule.agents

        unseated_students = [a for a in agents if not a.found_seat]

        if unseated_students:
            s = np.random.choice(unseated_students)
            s.step()
        else:
            return

        # update the theater plan for plotting
        offset = 0
        for col in range(self.plan.shape[1]):
            if col in np.cumsum(self.blocks)+np.arange(len(self.blocks)):
                offset += 1
                continue

            self.plan[:, col] = self.theater[:, col - offset]

        # self.print_theater()

    def fill_graph(self, sparsity):
        '''
        Fill out the friendship graph

        Args:
            sparsity: Percentage of matrix to filled. Low numbers coorespond to
                      a spare graph
        '''

        density = 0
        # Simple, random graph based on some sparsity
        s = self.friendships.shape[0]
        while(density < sparsity):
            x = np.random.randint(s)
            y = np.random.randint(s)
            self.friendships[x][y] = 1
            self.friendships[y][x] = 1
            density = np.sum(self.friendships) / (s**2)

        print('Friendships filled')
        # print(self.friendships)

    def find_accesible_seats(self):
        ''' Returns a list of seats avaliable without pushing past someone '''

        # flood fill to find accesible seats
        acc_seats = []
        ranges_right = list(np.cumsum(self.blocks))
        ranges_left = [0] + ranges_right[:-1]

        for i in range(self.num_of_rows):
            row = self.theater[i, :]

            # check right of paths:
            for j in range(self.paths):
                for x in np.arange(ranges_right[j], ranges_right[j+1]):
                    seat = row[x]
                    if seat == 0:
                        if (x, i) not in acc_seats:
                            acc_seats.append((x, i))
                    else:
                        break

            # check left of paths:
            for j in range(self.paths):
                for x in np.arange(ranges_left[j+1], ranges_left[j], -1) - 1:
                    seat = row[x]
                    if seat == 0:
                        if (x, i) not in acc_seats:
                            acc_seats.append((x, i))
                    else:
                        break

        return acc_seats

    def find_seated_friends(self, id):
        '''
        Return a list of seats of seated friends.

        Args:
            id: Unique id of student
        '''

        # seated_friends = []
        #
        # for friend in [a for a in self.schedule.agents if
        #     self.friendships[id][a.id] == 1]:
        #
        #     if friend.found_seat:
        #         seated_friends.append(friend.pos)
        #
        # return seated_friends

        # epic one-liner :D
        return [friend.pos for friend in self.schedule.agents if (self.friendships[id][friend.id] == 1) and friend.found_seat]

    def print_theater(self):
        ''' Print theater in ASCII '''
        p_ids = list(np.cumsum(self.blocks))
        for i in np.arange(self.num_of_rows, 0, -1) - 1:
            line = '|'
            row_data = self.theater[i]
            for j, seat in enumerate(row_data):
                if j in p_ids:
                    line += ' '

                line += '_' if seat == 0 else 'o'

            if self.blocks[-1] == 0:
                line += ' '

            print(line + '|')
        print('\n')
