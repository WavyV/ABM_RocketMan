import random

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agent import ClassroomAgent


# Convention: coordinates are column first, row second: (x, y).
# (0, 6) is the leftmost column and seventh row from the back.
# Seats with large y values are closer to the front.


class ClassroomModel(Model):
    """A classroom model containing agents.

    Args:
        entrances: list, of (x, y) tuples indicating positions of entrances.
        seats: matrix, of size grid_height X grid_width, and of type boolean
            indicating whether a seat exists at a position or not. All seats
            are assumed to be facing the front.
    """
    def __init__(self, max_agents, grid_width, grid_height, entrances, seats):
        self.running = True  # Required by server for some reason :s
        self.max_agents = max_agents
        self.entrances = entrances
        self.seats = seats
        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        self.schedule = RandomActivation(self)
        self.agents_in_classroom = set()  # Set of agent IDs.
        self.datacollector = DataCollector(
            model_reporters={"grid": lambda model: model.grid})

    def num_agents(self):
        """The amount of agents currently in the classroom."""
        return len(self.agents_in_classroom)

    def next_agent_id(self):
        """The next agent's unique ID.

        Runs in Θ(num_agents / 2) but could easily be done faster.
        """
        while True:
            agent_id = random.randint(0, self.max_agents)
            if agent_id not in self.agents_in_classroom:
                return agent_id

    def enter_agent(self, entrance):
        """A new agent enters the classroom at given entrance."""
        agent_id = self.next_agent_id()
        agent = ClassroomAgent(agent_id, self, entrance)
        self.schedule.add(agent)
        self.grid.place_agent(agent, entrance)

    def enter_agent_maybe(self):
        """A new agent may with some chance enter at a random entrance."""
        if random.random() < 0.4 and self.num_agents() < self.max_agents:
            entrance = random.choice(self.entrances)
            print("Agent entered at: {0}".format(entrance))
            self.enter_agent(entrance)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()  # Run step of all agents in classroom.
        self.enter_agent_maybe()
