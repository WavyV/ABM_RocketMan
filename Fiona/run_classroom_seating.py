#!/usr/bin/python3

from classroom_seating import *
import matplotlib.pyplot as plt

plot_all_steps = True
num_iterations = 5
max_num_agents = 50
width = 25
height = 25

""" Run the model """
model = ClassroomModel(max_num_agents, width, height)

for i in range(num_iterations):
    model.step()

    # visualization
    if plot_all_steps or i == num_iterations-1:
        seating_distr = np.zeros((model.grid.width, model.grid.height))
        for cell in model.grid.coord_iter():
            content, x, y = cell
            for agent in content:
                if type(agent) is Seat:
                    seating_distr[x,y] += 1
                if type(agent) is Student:
                    seating_distr[x,y] += 1

        for pos in model.classroom.entrances:
            seating_distr[pos[0],pos[1]] = -1

    plt.imshow(seating_distr, interpolation=None)
    plt.show()
