#!/usr/bin/python3

from classroom_seating import *
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
import pickle
import sys
from matplotlib.text import OffsetFrom

MODEL_DATA_PATH = "_model_data.json"
NUM_ITERATIONS = 300

"""
Run simulations of the ClassroomModel and visualize the seating process

Usage:
    python3 run_classroom_seating.py generate
    python3 run_classroom_seating.py animate
"""


"""
Determine relevant properties of the current model state and create an image representation

Args:
    model: the classroom model to analyse
    utilities: list of booleans [location, sociability, friendship] specifying which utilities are used in the model

Returns:
    image: 2D array representing the classroom.
                - The level of seat blocking is represented by decreasing negative values (the smaller the costlier to reach the seat)
                - The level of "happiness" of each student is represented by positive values (the larger the happier)
    info: 3D array comprising the 4 matrices 'seat_utilities', 'student_IDs', 'student_sociabilities' and 'initial_happiness'
"""
def get_model_state(model, utilities):
    image = -8*np.ones((model.grid.height, model.grid.width))
    info = np.zeros((model.grid.height, model.grid.width, 4))

    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat:
                # save seat utility
                info[y,x,0] = model.classroom.seats[x,y]
                if agent.student is None:
                    # seat is available. Determine level of blocking
                    image[y,x] = -10-agent.get_stand_up_cost()
                else:
                    # seat is occupied. Determine happiness of the student (depending on the utility component used in the given model)
                    image[y,x] = 10

                    if utilities[0]:
                        image[y,x] += agent.get_position_utility()
                    if utilities[1] and utilities[2]:
                        image[y,x] += agent.get_social_utility(agent.student)
                    else:
                        if utilities[1]:
                            image[y,x] += agent.get_social_utility(agent.student, use_friendship=False)
                        if utilities[2]:
                            image[y,x] += agent.get_social_utility(agent.student, use_sociability=False)

                    # save student's properties
                    info[y,x,1] = agent.student.unique_id
                    info[y,x,2] = agent.student.sociability
                    info[y,x,3] = agent.student.initial_happiness

    for pos in model.classroom.entrances:
        image[pos[1],pos[0]] = -20

    return image, info


"""
Set the parameters
"""
seed = 0
num_iterations = 300

# The classroom layout
blocks = [10, 15, 10]
num_rows = 20

# The social network of Students
cliques = 60
clique_size = 6
max_num_agents = cliques * clique_size
prob_linked_cliques = 0.3
social_network = nx.to_numpy_matrix(nx.relaxed_caveman_graph(cliques, clique_size, prob_linked_cliques, seed))


"""
Initialize the models
"""
model_random = ClassroomModel(max_num_agents, num_rows, blocks, seed, False, False)
model_sociability = ClassroomModel(max_num_agents, num_rows, blocks, seed, True, False)
model_friendship = ClassroomModel(max_num_agents, num_rows, blocks, seed, True, False, social_network)

models = [model_random, model_sociability, model_friendship]
model_names = ["blocking", "blocking + sociability", "blocking + sociability + friendship"]
# specify which model uses which utilities (location, sociability, friendship)
model_utilities = [[True,False,False],[True,True,True],[True,True,True]]


"""
Initialize the plots
"""
fig, axs = plt.subplots(1, len(models), figsize=(5*len(models),5))

min_value = -20
max_value = 20

images = []
model_data = []
annotes = []

for i, ax in enumerate(fig.axes):

    image, info = get_seat_blocking(models[i], model_utilities[i])
    images.append(ax.imshow(image, vmin=min_value, vmax=max_value, cmap = "RdYlGn", interpolation=None))
    ax.axis("off")
    ax.set_title(model_names[i])

    model_data.append(info)

    # initialize annotations
    helper = ax.annotate("", xy=(0.5, 0), xycoords="axes fraction",
                  va="bottom", ha="center")
    offset_from = OffsetFrom(helper, (0.5, 0))
    a = ax.annotate("seat", xy=(0,0), xycoords="data",
                  xytext=(0, -10), textcoords=offset_from,
                  va="top", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"), alpha=1)
    a.set_visible(False)
    annotes.append(a)


"""
Mouse cursor interactivity
"""
def hover(event):
    for i, ax in enumerate(fig.axes):
        a = annotes[i]
        if event.inaxes == ax:
            cond, ind = images[i].contains(event)
            if cond:
                x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                value = int(images[i].get_array()[y][x])
                seat_utility, student_id, sociability, initial_happiness = model_data[i][y,x]

                text = ""
                if value == -8:
                    text = "AISLE"
                elif value == -20:
                    text = "DOOR"
                elif value <= -10:
                    text = "EMPTY SEAT\nSeat utility: {}\nAccessibility: {}".format(seat_utility, value + 10)
                elif value >= 0:
                    text = "FILLED SEAT\n {} {:.2f} \n {} {} \n {} {:.2f} \n {} {:.2f}".format("Seat utility:", seat_utility,
                                "Student sociability:", int(sociability), "Initial happiness (t = {}):".format(int(student_id)),
                                initial_happiness, "Current happiness:", value - 10)

                a.set_text(text)
                a.xy = (x, y)
                a.set_visible(True)
                fig.canvas.draw_idle()

        else:
            a.set_visible(False)
            fig.canvas.draw_idle()



"""
Returns images for a given iteration, from the given model data.
"""
def iteration_images(iteration, all_model_states):
    fig.canvas.set_window_title("Iteration: {}".format(iteration))
    for i in range(len(models)):
        image, info = all_model_states[i][iteration]
        images[i].set_data(image)
        model_data[i] = info
    return tuple(images)


"""
Animate the models using data loaded from a file.
"""
def animate_models(num_iterations):
    # First load the previously generated model data.
    with open(MODEL_DATA_PATH, "rb") as f:
        all_model_states = pickle.load(f)
    # Initial animation state.
    iteration = 0
    running = True

    # Manages the logic deciding what the next iteration is.
    # Returns a tuple of images for the decided upon iteration.
    def next_iteration(_):
        nonlocal iteration
        if running:
            iteration += 1
        iteration = max(0, min(iteration, num_iterations - 1))
        return iteration_images(iteration, all_model_states)

    # Alter animation procedure based on user input.
    def key_press_handler(event):
        nonlocal iteration
        nonlocal running
        # Pause/resume animation on "p" key.
        if event.key == "p":
            running ^= True
        # Move to last frame on "e" key.
        elif event.key == "e":
            iteration = num_iterations - 1
        # Move to initial frame on "0" key.
        elif event.key == "0":
            iteration = -1
        # Move left on "←" key.
        elif event.key == "left":
            iteration -= 1
        # Move right on "→" key.
        elif event.key == "right":
            iteration += 1
        # Move 10 left on "a" key.
        elif event.key == "a":
            iteration -= 10
        # Move 10 right on "d" key.
        elif event.key == "d":
            iteration += 10

    fig.canvas.mpl_connect("key_press_event", key_press_handler)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    anim = animation.FuncAnimation(fig, next_iteration, frames=None,
                                   interval=300)
    fig.tight_layout()
    plt.show()


"""
Generate the necessary model state for each animation.
"""
def generate_data(num_iterations):
    # Save model state for each model for each step
    all_model_states = []
    for i in range(len(models)):
        print("Model {0}".format(i + 1))
        model_states = []
        for iteration in range(num_iterations):
            print("Iteration {0}".format(iteration + 1))
            models[i].step()
            model_states.append(get_seat_blocking(models[i], model_utilities[i]))
        all_model_states.append(model_states)
    # Save all model state data.
    with open(MODEL_DATA_PATH, "wb") as f:
        pickle.dump(all_model_states, f)


if __name__ == "__main__":
    if sys.argv[1] == "animate":
        animate_models(NUM_ITERATIONS)
    elif sys.argv[1] == "generate":
        generate_data(NUM_ITERATIONS)
