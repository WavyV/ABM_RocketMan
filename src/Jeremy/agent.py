import random

from mesa import Agent


class ClassroomAgent(Agent):
    """An agent in the classroom."""
    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        self.pos = position
        self.time_in_classroom = 0

    def step(self):
        """One step for the agent."""
        self.move()
        self.time_in_classroom += 1

    def possible_steps(self):
        """Free cells surrounding the agent."""
        neighbor_cells = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        return list(filter(self.can_move_to_neighbor_cell, neighbor_cells))

    def can_move_to_neighbor_cell(self, neighbor):
        """Whether a neighor cell can be moved to from current position."""
        x, y = self.pos
        x_n, y_n = neighbor
        # A seat must be horizontally located (no climbing over seats).
        if self.model.seats[y_n][x_n] and y == y_n:
            return False
        return True

    def move(self):
        """Move to a neighbouring point on the grid."""
        possible_steps = self.possible_steps()
        self.model.grid.move_agent(self, random.choice(possible_steps))
