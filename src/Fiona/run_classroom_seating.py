#!/usr/bin/python3

from classroom_seating import *
import matplotlib.pyplot as plt
import networkx as nx

"""
Determine the seating distribution of a given model instance

Args:
    model: the classroom model to analyse

Returns:
    seating_distr: 2D array representing the classroom.
                    - 0 represents aisles
                    - 1 represents available seats
                    - 2 represents occupied seats
                    - 2 represents entrances
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
        seating_distr[pos[0],pos[1]] = -2

    return seating_distr

"""
Determine the "stand-up-cost" for all seats

Args:
    model: the classroom model to analyse

Returns:
    blocking: 2D array representing the classroom.
                    - 0 represents aisles
                    - 2 represents occupied seats
                    - negative values represent the "stand-up-cost"
                    (the more students block the way between aisle and seat, the smaller the value)
"""
def get_seat_blocking(model):
    blocking = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        content, x, y = cell
        for agent in content:
            if type(agent) is Seat and not agent.occupied:
                blocking[x,y] = -agent.get_stand_up_cost()
            else:
                blocking[x,y] = 2

    return blocking


"""
Run the model
"""
seed = 0

plot_all_steps = False
num_iterations = 200
max_num_agents = 400
width = 30
height = 20

cliques = 100
prob_linked_cliques = 0.3
social_network = nx.to_numpy_matrix(nx.relaxed_caveman_graph(cliques, int(max_num_agents/cliques), prob_linked_cliques, seed))

model_random = ClassroomModel(max_num_agents, height, width, seed, False)
model_sociability = ClassroomModel(max_num_agents, height, width, seed, True)
model_friendship = ClassroomModel(max_num_agents, height, width, seed, True, social_network)

for i in range(num_iterations):

    # advance all models
    model_random.step()
    model_sociability.step()
    model_friendship.step()

    # visualization
    if plot_all_steps or i == num_iterations-1:

        distr_random = get_seating_distribution(model_random).T
        blocking_random = get_seat_blocking(model_random).T

        distr_sociability = get_seating_distribution(model_sociability).T
        blocking_sociability = get_seat_blocking(model_sociability).T

        distr_friendship = get_seating_distribution(model_friendship).T
        blocking_friendship = get_seat_blocking(model_friendship).T

        min_value = np.min([blocking_random,blocking_sociability,blocking_friendship])
        max_value = np.max([blocking_random,blocking_sociability,blocking_friendship])

        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        ax1.axis('off')
        ax1.imshow(distr_random, vmin=min_value, vmax=max_value, interpolation=None)
        ax1.set_title("random")
        #ax1.set_ylabel("final seating distribution", rotation=0, size='large')

        ax2 = fig.add_subplot(2,3,4)
        ax2.axis('off')
        ax2.imshow(blocking_random, vmin=min_value, vmax=max_value, interpolation=None)
        #ax2.set_ylabel("row blocking", rotation=0, size='large')

        ax3 = fig.add_subplot(2,3,2)
        ax3.axis('off')
        ax3.imshow(distr_sociability, vmin=min_value, vmax=max_value, interpolation=None)
        ax3.set_title("sociability")

        ax4 = fig.add_subplot(2,3,5)
        ax4.axis('off')
        ax4.imshow(blocking_sociability, vmin=min_value, vmax=max_value, interpolation=None)

        ax5 = fig.add_subplot(2,3,3)
        ax5.axis('off')
        ax5.imshow(distr_friendship, vmin=min_value, vmax=max_value, interpolation=None)
        ax5.set_title("friendship")

        ax6 = fig.add_subplot(2,3,6)
        ax6.axis('off')
        ax6.imshow(blocking_friendship, vmin=min_value, vmax=max_value, interpolation=None)

        fig.tight_layout()

        plt.show()
