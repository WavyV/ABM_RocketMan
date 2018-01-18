#!/usr/bin/python3

from classroom_seating import *
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
import pickle
import sys

MODEL_DATA_PATH = "_model_data.json"
NUM_ITERATIONS = 300

"""
Run simulations of the ClassroomModel and visualize the seating process

Usage:
    python3 run_classroom_seating.py generate
    python3 run_classroom_seating.py animate
"""


"""
Determine the seating distribution of a given model instance

Args:
    model: the classroom model to analyse

Returns: seating_distr: 2D array representing the classroom with aisles,
    entrances, seats and students """
def get_seating_distribution(model):
    seating_distr = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat:
                if agent.student is None:
                    # seat is available
                    seating_distr[x,y] = -5
                else:
                    # seat is occupied. Determine happiness of the student
                    seating_distr[x,y] = 5 + agent.get_total_utility(agent.student)

                    #seating_distr[x,y] = 2

    for pos in model.classroom.entrances:
        seating_distr[pos[0],pos[1]] = -15

    return seating_distr


"""
Determine the "stand-up-cost" for all seats

Args:
    model: the classroom model to analyse
    utilities: list of booleans [location, sociability, friendship] specifying
        which utilities are used in the model

Returns:
    blocking: 2D array representing the classroom.
        - The level of seat blocking is represented by decreasing negative
          values (the smaller the costlier to reach the seat)
        - The level of "happiness" of each student is represented by
          positive values (the larger the happier) """
def get_seat_blocking(model, utilities):
    blocking = -8*np.ones((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat:
                if agent.student is None:
                    # seat is available. Determine level of blocking
                    blocking[x,y] = -10-agent.get_stand_up_cost()
                else:
                    # seat is occupied. Determine happiness of the student
                    # (depending on the utility component used in the given
                    # model)
                    blocking[x,y] = 10

                    if utilities[0]:
                        blocking[x,y] += agent.get_position_utility()
                    if utilities[1] and utilities[2]:
                        blocking[x,y] += agent.get_social_utility(agent.student)
                    else:
                        if utilities[1]:
                            blocking[x,y] += agent.get_social_utility(agent.student, use_friendship=False)
                        if utilities[2]:
                            blocking[x,y] += agent.get_social_utility(agent.student, use_sociability=False)

    for pos in model.classroom.entrances:
        blocking[pos[0],pos[1]] = -20

    return blocking


"""
Set the parameters
"""
seed = 0

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
#min_value = -max(blocks)
#max_value = 5
min_value = -20
max_value = 20

images = []

for i, ax in enumerate(fig.axes):

    model_state = get_seat_blocking(models[i], model_utilities[i]).T
    images.append(ax.imshow(model_state, vmin=min_value, vmax=max_value,
                            cmap="RdYlGn", interpolation=None))
    ax.axis("off")
    ax.set_title(model_names[i])


"""
Returns images for a given iteration, from the given model data.
"""
def iteration_images(iteration, all_model_states):
    fig.canvas.set_window_title("Iteration: {}".format(iteration))
    for i in range(len(models)):
        model_state = all_model_states[i][iteration]
        images[i].set_data(model_state)
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
            model_states.append(get_seat_blocking(models[i], model_utilities[i]).T)
        all_model_states.append(model_states)
    # Save all model state data.
    with open(MODEL_DATA_PATH, "wb") as f:
        pickle.dump(all_model_states, f)


if __name__ == "__main__":
    if sys.argv[1] == "animate":
        animate_models(NUM_ITERATIONS)
    elif sys.argv[1] == "generate":
        generate_data(NUM_ITERATIONS)
