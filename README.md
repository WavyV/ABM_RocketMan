# ABM_RocketMan
Agent-Based Modelling Course Project

## Code Layout

Initial Mesa based model supporting a grid with entrances, a seat layout and
visualization. Agents are created stochastically at each time step at one of the
given entrances. Agents can move to cells without seats and horizontally in a
seat row.

There is no smart agent behaviour but I believe this code forms a good basis for
expansion since the code is modular and extensible. It will allow us to now
separate coding work and work on tasks independently.

## Visualization

The visualization supported by Mesa is very simple. I havn't seen how to portray
cell backgrounds independently. I have only seen how to portray an agent. I
would like to investigate how to portray cells such as entrances and seats
differently from empty floor space, if this isn't supported by Mesa it could
mean a useful PR for the community.
