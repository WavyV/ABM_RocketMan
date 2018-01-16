#!/usr/bin/python3

from classroom_seating import *
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation

"""
Run simulations of the ClassroomModel and visualize the seating process

Usage: python3 run_classroom_seating.py
"""


"""
Determine the seating distribution of a given model instance

Args:
    model: the classroom model to analyse

Returns:
    seating_distr: 2D array representing the classroom with aisles, entrances, seats and students
"""
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
    utilities: list of booleans [location, sociability, friendship] specifying which utilities are used in the model

Returns:
    blocking: 2D array representing the classroom.
                - The level of seat blocking is represented by decreasing negative values (the smaller the costlier to reach the seat)
                - The level of "happiness" of each student is represented by positive values (the larger the happier)
"""
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
                    # seat is occupied. Determine happiness of the student (depending on the utility component used in the given model)
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
#min_value = -max(blocks)
#max_value = 5
min_value = -20
max_value = 20

images = []

for i, ax in enumerate(fig.axes):

    model_state = get_seat_blocking(models[i], model_utilities[i]).T
    images.append(ax.imshow(model_state, vmin=min_value, vmax=max_value, cmap = "RdYlGn", interpolation=None))
    ax.axis("off")
    ax.set_title(model_names[i])


"""
Run and visualize the models
"""
def animate(i):
    # advance all models
    for i in range(len(models)):
        models[i].step()
        model_state = get_seat_blocking(models[i], model_utilities[i]).T
        images[i].set_data(model_state)

    return tuple(images)

anim = animation.FuncAnimation(fig, animate, frames=num_iterations, interval=300, repeat=False)

fig.tight_layout()
plt.show()
