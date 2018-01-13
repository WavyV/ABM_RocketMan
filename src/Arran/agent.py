from mesa import Agent
import numpy as np

def pickRandom(x):
    '''
    Pick an integer up to x (but not including), with bias toward smaller numbers
    '''
    y = np.arange(x, 0, -1)
    p = y / sum(y)
    return np.random.choice(y, p=p) - 1

class Student(Agent):
    """ A student with seating strategies """
    goal = None

    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.found_seat = False


    def step(self):
        if self.found_seat:
            return

        self.find_goal()
        if self.goal != None:
            # self.find_goal()
            self.model.grid.move_agent(self, self.goal)
            self.model.theater[self.goal[1]][self.goal[0]] = 1
            print('Taken seat {}'.format(self.goal))

        # Move toward goal

        self.found_seat = True


        # if not self.found_row:
        #     if np.random.random < 0.25:
        #         self.found_row = True
        #     else:
        #         move_along_path()
        # else:
        #     move_along_row()

        # decide to move up path or into a row

            # move along row


        # path = pickRandom(self.model.paths)
        # if self.door >= 3:
        #     # incase on opposite side flip side
        #     path = self.model.paths - path - 1
        #
        # # TODO: make this a poisson distribution that extends multiple paths?
        # row = pickRandom(self.model.num_of_rows)
        # if (self.door == 2) or (self.door == 3):
        #     # incase at top of theater start from top
        #     row = self.model.num_of_rows - row - 1
        #
        # side = np.random.randint(2)
        # if self.model.rows[path] == 0:
        #     side = 1
        # elif self.model.rows[path + 1] == 0:
        #     side = 0
        #
        # row_seats = self.model.theater[row, path + side]
        #
        # # count avaliable seats:
        #
        #
        # self.model.theater[row][col] = 1
        # self.found_seat = True

    def find_goal(self):
        # for now, pick a random empty, accesible seat
        acc_seats = self.model.find_accesible_seats()

        if len(acc_seats) > 0:
            self.goal = acc_seats[np.random.randint(len(acc_seats))]
        else:
            print('No seats left :(')
            return
