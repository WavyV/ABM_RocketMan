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
    seating_distr: 2D array representing the classroom.
                    - 0 represents aisles
                    - 1 represents available seats
                    - 2 represents occupied seats
                    - -1 represents entrances
"""
def get_seating_distribution(model):
    seating_distr = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat and not agent.occupied:
                seating_distr[x,y] = 1
            else:
                seating_distr[x,y] = 2

    for pos in model.classroom.entrances:
        seating_distr[pos[0],pos[1]] = -1

    return seating_distr

"""
Determine the "stand-up-cost" for all seats

Args:
    model: the classroom model to analyse

Returns:
    blocking: 2D array representing the classroom.
                    - 3 represents aisles
                    - 5 represents occupied seats
                    - negative values represent the "stand-up-cost"
                    (the more students block the way between aisle and seat, the smaller the value)
                    - the smallest negative value represents the entrances
"""
def get_seat_blocking(model):
    blocking = 3*np.ones((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat and not agent.occupied:
                blocking[x,y] = -agent.get_stand_up_cost()
            else:
                blocking[x,y] = 5

    for pos in model.classroom.entrances:
        blocking[pos[0],pos[1]] = -max(blocks)

    return blocking


"""
Set the parameters
"""
seed = 0
num_iterations = 200

# The classroom layout
blocks = [0, 10, 20, 10]
num_rows = 20

# The social network of Students
cliques = 75
clique_size = 4
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


"""
Initialize the plots
"""
fig, axs = plt.subplots(1, len(models), figsize=(5*len(models),5))
min_value = -max(blocks)
max_value = 5

images = []

for i, ax in enumerate(fig.axes):

    model_state = get_seat_blocking(models[i]).T
    images.append(ax.imshow(model_state, vmin=min_value, vmax=max_value, interpolation=None))
    ax.axis("off")
    ax.set_title(model_names[i])


"""
Run and visualize the models
"""
def animate(i):
    # advance all models
    for i in range(len(models)):
        models[i].step()
        model_state = get_seat_blocking(models[i]).T
        images[i].set_data(model_state)

    return tuple(images)

anim = animation.FuncAnimation(fig, animate, frames=num_iterations, interval=500, repeat=False)

fig.tight_layout()
plt.show()
