import json

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from model import ClassroomModel


max_agents = 4
seats = [
    [False, False, False, False, False, False, False],
    [False, True, True, True, True, True, False],
    [False, True, True, True, True, True, False],
    [False, True, True, True, True, True, False],
]
grid_width = len(seats[0])
grid_height = len(seats)
entrances = [
    (0, 0),
    (grid_width - 1, 0)
]
num_steps = 20


def agent_portrayal(agent):
    """Given an agent returns how she should be visualized."""
    return {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "Color": "red",
        "r": 0.5
    }


grid = CanvasGrid(agent_portrayal, grid_width, grid_height, 500, 500)
parameters = {
    "max_agents": max_agents,
    "grid_width": grid_width,
    "grid_height": grid_height,
    "entrances": entrances,
    "seats": seats,
}
print("Parameters:\n{0}".format(
    json.dumps(parameters, indent=4, sort_keys=True)))
server = ModularServer(
    ClassroomModel,
    [grid],
    "Classroom Model",
    parameters
)
