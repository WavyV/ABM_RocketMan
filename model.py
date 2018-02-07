from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom
from social import network

"""
This is the main modeling module, providing classes for
    - the general classroom model
    - students
    - seats
    - classroom design

General usage:
    - initialize the model: m = ClassroomModel(classroom_design, coefs=[0.25,0.25,0.25,0.25], sociability_sequence, degree_sequence, seed=0):
    - run the model for one time step: m.step()
    - get the current seating distribution: output = m.get_binary_model_state()

"""



class Student():

    """
    Create a student with individual characteristics

    Args:
        unique_id: student identifier
        model: the associated ClassroomModel
        sociability: willingness to sit next to unknown people (-1: unsociable, 0: indifferent, 1: sociable)
    """
    def __init__(self, unique_id, model, sociability=0):
        self.model = model
        self.unique_id = unique_id
        self.sociability = sociability

        # initial state of the student
        self.seated = False
        self.initial_happiness = 0

        # Currently not used, but here incase of future study
        self.will_to_change_seat = False
        self.moving_threshold = 1
        self.moving_prob = 0.05



    """
    The seat selection procedure

    Args:
        seat_pos: predetermined position of the seat to choose. If this parameter is specified utilities are ignored.
        old_seat: current seat of the student making the seating decision. If this parameter is specified the student is enabled to move to a seat with higher utility than his current one.
    """
    def choose_seat(self, seat_pos=None, old_seat=None):

        seat_choice = None

        if seat_pos is None:
            # determine all possible seats to choose from
            seat_options = self.model.empty_seats

            if len(seat_options) == 0:
                print("No empty seats!")

            else:
                if self.model.random_seat_choice:
                    # pick one randomly
                    seat_choice = self.model.rand.choice(seat_options)
                else:
                    if old_seat is None:
                        seat_utilities = [seat.get_total_utility(self) for seat in seat_options]

                        if self.model.deterministic_choice:
                            # always choose among the seats with highest utility
                            seat_choice = self.model.rand.choice(np.array(seat_options)[np.where(seat_utilities == np.max(seat_utilities))])

                        else:

                            # determine the best 'seat_fraction' (e.g. 50%) of all available seats
                            seat_subset, utility_subset = [], []
                            for i in range(int(self.model.seat_fraction * len(seat_options))):
                                index = np.argmax(seat_utilities)
                                seat_subset.append(seat_options[index])
                                utility_subset.append(seat_utilities[index])
                                seat_utilities[index] = 0
                            sum_utilities = sum(utility_subset)

                            if sum_utilities > 0:
                            # convert utilities into probabilities and choose seat based on the resulting probability distribution
                                utility_subset = [s/sum_utilities for s in utility_subset]
                                seat_choice = self.model.rand.choice(seat_subset, p=utility_subset)

                            else:
                                # if all utilities are zero, choose the seat randomly
                                seat_choice = self.model.rand.choice(seat_options)

                    else:
                        # only happens if 'will_to_change_seat' is enabled!

                        # pick seat with highest utiltiy (if multiple seats are optimal, choose one of them randomly)
                        # include cost to get away from the old seat
                        seat_utilities = [seat.get_total_utility(self) - old_seat.get_stand_up_cost() for seat in seat_options]

                        if np.max(seat_utilities) - old_seat.get_total_utility(self) > self.moving_threshold:
                            # if the difference in utility between the current seat and another available seat exceeds the threshold, move to one of the optimal ones
                            seat_choice = self.model.rand.choice(np.array(seat_options)[np.where(seat_utilities == np.max(seat_utilities))])

        else:
            # get the seat object at the predetermined position
            seat = self.model.seats[seat_pos]
            if seat.student is None:
                seat_choice = seat


        if seat_choice is not None:
            # seat has been selected
            if old_seat is not None:
                # make seat available again
                old_seat.student = None
                self.empty_seats.append(old_seat)

            # move to the selected seat
            seat_choice.student = self
            self.initial_happiness = seat_choice.get_happiness(self)
            self.seated = True

            # update the accessibility of all seats in the row
            for s in self.model.seats[:, seat_choice.pos[1]]:
                if type(s) == Seat:
                    s.update_accessibility()

            self.model.empty_seats.remove(seat_choice)


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
            r = self.model.rand.uniform()
            if r < self.moving_prob:
                # compare current seat utility with all other available seats. If there is a much better one, move
                try:
                    seat = self.model.seats[self.pos]
                    self.choose_seat(seat)
                except:
                    return


class Seat():

    """
    Create a seat

    Args:
        model: the associated ClassroomModel
        pos: the (x, y) coordinates of the seat
    """
    def __init__(self, model, pos):

        self.model = model
        self.student = None
        self.accessibility = 1
        self.pos = pos

        x, y = pos

        # find the distances to the left and right aisles, for accessibility
        if x < model.classroom.aisles_x[0]:
            # then no aisle to the left
            self.row_left = None
        else:
            left_aisle = max([a for a in model.classroom.aisles_x if a < x])
            self.row_left = [(i, y) for i in range(left_aisle+1, x)]

        if x > model.classroom.aisles_x[-1]:
            # then no aisle to the right
            self.row_right = None
        else:
            right_aisle = min([a for a in model.classroom.aisles_x if a > x])
            self.row_right = [(i, y) for i in range(x+1, right_aisle)]

    """
    Get the position utility of the seat, based on its location in the Classroom

    Returns:
        pos_utility: utility in range [0,1]
    """
    def get_position_utility(self):
        return self.model.classroom.pos_utilities[self.pos]

    """
    Get the accessibility of the seat based on the number of Students between the seat and the aisle.
    The student count for the left and right side are compared and the minimum is used.
    This number is normalized by the maximal possible number of students to pass (dependent on the classroom design)
    and then substracted from one in order to obtain values between 0 (number of students to be passed is maximal)
    and 1 (no students to be passed).

    Returns:
        u_accessibility: utiltiy in range [0,1]
    """
    def update_accessibility(self):
        x, y = self.pos

        count_left, count_right = 0, 0

        if self.row_left != None:
            for cell in self.row_left:
                if self.model.seats[cell].student:
                    count_left += 1
        else:
            count_left = np.infty

        if self.row_right != None:
            for cell in self.row_right:
                if self.model.seats[cell].student:
                    count_right += 1
        else:
            count_right = np.infty

        self.accessibility = 1 - min(count_right, count_left)/float(self.model.classroom.max_pass)


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

        # assure that size_x and size_y are odd, so that the neighborhood has one central seat
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

                if (coords[0] < 0) or (coords[0] > self.model.classroom.width) or (coords[1] < 0) or (coords[1] > self.model.classroom.num_rows):
                    continue

                else:
                    try:
                        seat = self.model.seats[coords]
                        neighborhood[i, j] = seat.student.unique_id
                    except:
                        continue

        return neighborhood


    """
    Get the social utility of the Seat (including both friendship and sociability component).
    Higher values represent higher attractivity.
    Values are based on the social network connections to neighboring Students,
    and the individual sociability of the Student making the decision:
        - The more friends are sitting closeby, the higher the friendship utility.
        - The more sociable the student, the higher the utility in case of unfamiliar neighbors

    Args:
        student: the student making the seating choice

    Returns:
        u_friendship: friendship component of the social utility (in range [0,1])
        u_sociability: sociability component of the social utility (in range [0,1])
    """
    def get_social_utility(self, student):

        interaction_x, interaction_y = self.model.friendship_interaction_matrix.shape

        neighborhood = self.get_neighborhood(interaction_x, interaction_y)
        u_friendship = 0
        u_sociability = 0

        for x in range(interaction_x):
            for y in range(interaction_y):

                # if there is a student check friendship
                if neighborhood[x,y] >= 0:
                    friendship = self.model.social_network[int(student.unique_id), int(neighborhood[x,y])]

                    u_friendship += self.model.friendship_interaction_matrix[x, y] * friendship

                    # if neighbouring seat is occupied by a student that is not a friend,
                    # determine the sociability component based on the student's sociability attribute
                    if friendship == 0:
                        u_sociability += self.model.sociability_interaction_matrix[x,y] * student.sociability

        # scale the final sociability term to range [0,1]
        s_min, s_max = self.model.sociability_range
        if s_max > s_min:
            u_sociability = max(0, u_sociability - s_min) / (s_max - s_min)
        else:
            u_sociability = s_min

        return u_friendship, u_sociability

    """
    Get the overall utility of the Seat as a linear combination of position, friendship, sociability and accessibility components

    Returns:
        total_utility: high values represent high attractivity of the seat
    """
    def get_total_utility(self, student):

        friendship_component, sociability_component = self.get_social_utility(student)
        coef_p, coef_f, coef_s, coef_a = self.model.coefs
        total_utility = coef_p * self.get_position_utility() + coef_f * friendship_component + coef_s * sociability_component + coef_a * self.accessibility

        return total_utility

    """
    Get the student's happiness for this Seat as a linear combination of position, friendship and sociability components.
    It is assumed that the accessibility does not influence the happiness one the student is seated.

    Returns:
        happiness: total seat utility except for the accessibility component
    """
    def get_happiness(self, student):

        friendship_component, sociability_component = self.get_social_utility(student)
        coef_p, coef_f, coef_s, coef_a = self.model.coefs
        total_utility = coef_p * self.get_position_utility() + coef_f * friendship_component + coef_s * sociability_component

        return total_utility



class ClassroomModel():

    """
    Create a classroom model

    Args:
        classroom_design: instance of ClassroomDesign defining the layout and position-dependent seat utilities of the classroom
        coefs: list [coef_p, coef_f, coef_s, coef_a] defining the coefficients for the position, friendship, sociability and accessibility components in the utility function
        sociability_sequence: list of sociability values per student. Should be sampled from a probability distribution of the students' sociability attribute
        social_network: the social network to use. Overrides degree_sequence
        degree_sequence: list of friendship degrees per student. Used to create the underlying social network in form of a connectivity matrix (1 means friendship, 0 means indifference).
                If not given, a random social network (erdos renyi) is created
        seed: seed for the random number generation
        seat_fraction: fraction of available seats to be considered for seat choice
        deterministic_choice: boolean if students pick deterministically the seat with the highest utility, or if choice is probabilitstic.
    """
    def __init__(self, classroom_design, coefs=[0.25, 0.25, 0.25, 0.25],
                 sociability_sequence=None, social_network=None,
                 degree_sequence=None, seed=0,
                 seat_fraction=0.5, deterministic_choice=True, scale=True):
        self.rand = np.random.RandomState(seed)
        self.classroom = classroom_design
        self.seat_fraction = seat_fraction
        self.deterministic_choice = deterministic_choice
        self.empty_seats = []
        self.students = []
        self.model_states = []   # all simulated model states stored here
        self.im = None   # used to store the current image

        # referenced as x, y (i.e. column then row!)
        self.seats = np.empty((self.classroom.width, self.classroom.num_rows), dtype=Seat)

        # assure that the coefficients sum up to one
        if scale:
            self.coefs = [(c/sum(coefs) if sum(coefs) > 0 else 0) for c in coefs]
        else:
            self.coefs = coefs

        # if all utility components are set to zero, seat choices are completely random
        self.random_seat_choice = np.all(coefs == 0)

        # setup the social network
        if social_network is not None:
            self.social_network = social_network
            self.max_num_agents = social_network.shape[0]
        else:
            if degree_sequence is None:
                # create a random network
                self.max_num_agents = self.classroom.seat_count
                self.social_network = network.erdos_renyi(self.max_num_agents, 0.2)[0]
            else:
                self.max_num_agents = len(degree_sequence)
                self.rand.shuffle(degree_sequence)
                self.social_network = network.walts_graph(degree_sequence, plot=False)[0]

        # set up the sociabilities of the students
        if sociability_sequence is None:
            # default sociability values are sampled uniformly from [0,1]
            self.sociability_sequence = deque([self.rand.uniform(0, 1) for _ in range(self.max_num_agents)])
            self.sociability_range = (min(self.sociability_sequence),max(self.sociability_sequence))
        else:
            if len(sociability_sequence) == self.max_num_agents:
                self.rand.shuffle(sociability_sequence)
                self.sociability_sequence = deque(sociability_sequence)
                self.sociability_range = (min(self.sociability_sequence),max(self.sociability_sequence))
            else:
                raise ValueError("'sociability_sequence' and 'degree_sequence' must have same length")

        # Matrices that determine the importance of neighboring seats for the social utility.
        # They need to have the same shape.
        # Values should sum up to one so that the resulting friendship and sociability terms are within range [0,1].
        self.friendship_interaction_matrix = np.array([[0.5, 0, 0.5]]).T
        self.sociability_interaction_matrix = np.array([[0.5, 0, 0.5]]).T

        # initialize seats (leave aisles free)
        for x in range(self.classroom.width):
            if not x in self.classroom.aisles_x:
                for y in range(self.classroom.num_rows):
                    if not y in self.classroom.aisles_y:
                        # create new seat
                        seat = Seat(self, (x, y))
                        self.empty_seats.append(seat)
                        self.seats[x, y] = seat

        self.model_states.append(self.get_model_state())

    """
    Advance the model by one step. If the maximum student number is not reached yet, create a new student every tick.
    """
    def step(self):
        # as long as the max number of students is not reached, add a new one
        n = len(self.students)
        if n < self.max_num_agents:
            # create student
            try:
                sociability = self.sociability_sequence.popleft()
            except:
                sociability = 0

            student = Student(n, self, sociability)

            # add student and update
            self.students.append(student)
            student.step()

            self.model_states.append(self.get_model_state())


    """
    Advance the model by one step. If the maximum student number is not reached yet, create a new student and place him at the given position.

    Args:
        seat_pos: position at which the new student should be seated
    """
    def step_predetermined_seating(self, seat_pos):
        n = len(self.students)
        if n < self.max_num_agents:
            # if max student count is not reached, create student
            student = Student(n, self)
            self.students.append(student)

            # place new student at the predetermined seat
            student.choose_seat(seat_pos)

    """
    Returns the current model state, with information about each seat and
    student
    """
    def get_model_state(self):
        image = -0.8*np.ones((self.classroom.num_rows, self.classroom.width))
        info = np.zeros((self.classroom.num_rows, self.classroom.width, 4))

        for seat in self.seats.flat:
            if type(seat) != Seat:
                continue

            x, y = seat.pos

            info[y,x,0] = self.classroom.pos_utilities[x,y]
            if seat.student is None:
                # seat is available. Determine level of accessibility
                image[y,x] = -2 + seat.accessibility
            else:
                # seat is occupied. Set value based on the student's happiness
                image[y,x] = 1
                image[y,x] += seat.get_happiness(seat.student)

                # save student's properties
                info[y,x,1] = seat.student.unique_id
                info[y,x,2] = seat.student.sociability
                info[y,x,3] = seat.student.initial_happiness


        for pos in self.classroom.entrances:
            image[pos[1],pos[0]] = -3

        return (image, info)


    """
    Get the current seating distribution in the classroom. Ones represent students, zeros represent available seats.
    Aisles are stripped.

    Returns:
        model_state: binary matrix where each entry refers to a seat's state
    """
    def get_binary_model_state(self):
        model_state = np.zeros((self.classroom.width, self.classroom.num_rows), dtype=int)
        for seat in self.seats.flat:
            # model_state[student.pos] = 1
            if seat is not None and seat.student:
                model_state[seat.pos] = 1
        return self.remove_aisles(model_state).T

    def get_happiness_model_state(self):
        """Return a matrix of the happiness of each student at each seat."""
        model_state = np.zeros((self.classroom.width, self.classroom.num_rows))
        for seat in self.seats.flat:
            if seat is not None and seat.student:
                model_state[seat.pos] = seat.get_happiness(seat.student)
        return self.remove_aisles(model_state).T

    def remove_aisles(self, model_state):
        """Remove aisles from the given matrix with shape of this model."""
        # remove aisles from the matrix
        model_state = np.delete(model_state, self.classroom.aisles_x, axis=0)
        model_state = np.delete(model_state, self.classroom.aisles_y, axis=1)
        return model_state


    """
    Using matplotlib to draw the current state of the model.

    Args:
        fig: the pyplot figure.
        ax: the axes to draw the plot.
        interactive: whether to handle mouse events.
        state: which model state to use. -1 can be used to show last state.
    """
    def plot(self, fig, ax, interactive=False, state=-1):
        try:
            image, info = self.model_states[state]
        except:
            image, info = self.get_model_state()

        ax.clear()

        self.im = ax.imshow(image, vmin=-2, vmax=2, cmap='RdYlGn', interpolation=None)

        ax.axis('off')

        if state < 0:
            num_students = len(self.students) + state + 1
        else:
            num_students = min(state, len(self.students))
        ax.set_title('Classroom State ({} students)'.format(num_students))

        helper = ax.annotate("", xy=(0.5, 0), xycoords="axes fraction",
                             va="bottom", ha="center")
        offset_from = OffsetFrom(helper, (0.5, 0))
        a = ax.annotate("seat", xy=(0, 0), xycoords="data", xytext=(0, -10),
                        textcoords=offset_from, va="top", ha="center",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"), alpha=1)
        a.set_visible(False)

        def hover(event):
            if(event.inaxes):
                x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                value = self.im.get_array()[y][x]
                seat_utility, student_id, sociability, initial_happiness = (
                    info[y, x])

                text = ""
                if value == -0.8:
                    text = "AISLE"
                elif value == -3:
                    text = "DOOR"
                elif value <= -1:
                    text = "EMPTY SEAT\nSeat attractivness: {:.2f}\nAccessibility: {:.2f}".format(
                        seat_utility, value + 2)
                elif value >= 0:
                    text = "FILLED SEAT\n {} {:.2f} \n {} {:.2f} \n {} {:.2f} \n {} {:.2f}".format(
                        "Seat attractivness:", seat_utility,
                        "Student sociability:", sociability,
                        "Initial happiness (t = {}):".format(int(student_id)),
                        initial_happiness, "Current happiness:", value - 1)

                a.set_text(text)
                a.xy = (x, y)
                a.set_visible(True)
                fig.canvas.draw_idle()

            else:
                a.set_visible(False)
                fig.canvas.draw_idle()

        ax.set_xticks(np.arange(-0.5, image.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, image.shape[0]), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

        if interactive:
            cid = fig.canvas.mpl_connect('motion_notify_event', hover)

        fig.tight_layout(rect=[0, 0.2, 1, 1])


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
    def __init__(self, blocks=[6,14,0], num_rows=14, pos_utilities=None, entrances=None, aisles_y=None):

        self.width = sum(blocks) + len(blocks) - 1
        self.num_rows = num_rows

        # define vertical aisles
        self.aisles_x = []
        current_x = 0
        for b in range(len(blocks)-1):
            current_x += blocks[b]
            self.aisles_x.append(current_x)
            current_x += 1

        # define horizontal aisles
        if aisles_y is None:
            # default: one aisle in the front
            self.aisles_y = [0]
        else:
            self.aisles_y = aisles_y

        # define entrances
        if entrances is None:
            # default: two entrances at the front corners
            self.entrances = [((0, 0)),((self.width-1,0))]
        else:
            self.entrances = entrances

        # define utility/attractivity of each seat location
        # if the utility matrix does not include aisles, insert zeros at the respective positions

        if pos_utilities is not None and pos_utilities.shape == (self.width - len(self.aisles_x), num_rows - len(self.aisles_y)):
            for x in self.aisles_x:
                if x < pos_utilities.shape[0]:
                    pos_utilities = np.insert(pos_utilities, x, 0, axis=0)
                else:
                    pos_utilities = np.concatenate((pos_utilities, np.zeros((1,pos_utilities.shape[1]))), axis=0)
            for y in self.aisles_y:
                if y < pos_utilities.shape[1]:
                    pos_utilities = np.insert(pos_utilities, y, 0, axis=1)
                else:
                    pos_utilities = np.concatenate((pos_utilities, np.zeros((pos_utilities.shape[0],1))), axis=1)

        if pos_utilities is not None and pos_utilities.shape == (self.width, num_rows) and np.max(pos_utilities) > 0:
            # scale the given attractivity weights to assure values in range [0,1]
            self.pos_utilities = pos_utilities/np.max(pos_utilities)
        else:
            print("Positional Utilities set to zeros...")
            self.pos_utilities = np.zeros((self.width, num_rows))

        # determine the maximal number of seats to be passed in order to get to a seat
        max_pass_per_block = []
        for i,b in enumerate(blocks):
            if i > 0 and i < len(blocks)-1:
                # for inner blocks with aisles on the left and right, only half of the row needs to be passed
                max_pass_per_block.append(int((b-1)/2))
            else:
                # for outer blocks with access to only one aisle, the entire row needs to be passed
                max_pass_per_block.append(b-1)
        self.max_pass = max(max_pass_per_block)

        # determine the total number of seats
        self.seat_count = (self.width - len(self.aisles_x)) * (self.num_rows - len(self.aisles_y))
