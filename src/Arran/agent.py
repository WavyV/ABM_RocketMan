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
    ''' A student with seating strategies '''

    goal = None

    def __init__(self, pos, id, model):
        '''
        Create new Student agent.

        Args:
            pos: Initial position
            id: Unique identification number
            model: The TheaterModel this agent belongs to
        '''
        super().__init__(pos, model)
        self.id = id
        self.found_seat = False

    def step(self):
        ''' Progress this Student one step '''

        if self.found_seat:
            return

        self.find_goal()
        if self.goal != None:
            self.model.grid.move_agent(self, self.goal)
            self.model.theater[self.goal[1]][self.goal[0]] = 1
            print('Taken seat {}'.format(self.goal))

            self.model.seated_students.append(self)
            self.found_seat = True

    def find_goal(self):
        ''' Find a seat '''

        # picks the first accesible seat next to friend, otherwise random
        acc_seats = self.model.find_accesible_seats()
        friend_seats = self.model.find_seated_friends(self.id)

        if len(acc_seats) > 0:
            if len(friend_seats) > 0:
                # check if there are any accesible seats next to friend
                for x, y in friend_seats:
                    # check left seat
                    if (x-1, y) in acc_seats:
                        self.goal = (x-1, y)
                        print('Next to friend :)')
                        return

                    # check right seat
                    if (x+1, y) in acc_seats:
                        print('Next to friend :)')
                        self.goal = (x+1, y)
                        return
                else:
                    print('Can\'t sit next to friend :(')
                    self.goal = acc_seats[np.random.randint(len(acc_seats))]

            else:
                print('No seated friends :(')
                self.goal = acc_seats[np.random.randint(len(acc_seats))]

        else:
            print('No seats left :(')
            return

    def check_next_to_friend(self):
        ''' Returns the number of friends student is next to (0, 1 or 2) '''
