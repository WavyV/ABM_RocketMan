#!/usr/bin/python3

from classroom_seating import *
import matplotlib.pyplot as plt

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
plot_all_steps = False
num_iterations = 150
max_num_agents = 200
width = 30
height = 20

model = ClassroomModel(max_num_agents, height, width)

for i in range(num_iterations):
    model.step()

    # visualization
    if plot_all_steps or i == num_iterations-1:

        distr = get_seating_distribution(model).T
        blocking = get_seat_blocking(model).T

        plt.subplot(211)
        plt.imshow(distr, vmin=np.min(blocking), vmax=np.max(blocking), interpolation=None)

        plt.subplot(212)
        plt.imshow(blocking, vmin=np.min(blocking), vmax=np.max(blocking), interpolation=None)

        plt.show()
