import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

## create graph
G = nx.random_partition_graph([10,10,10], 0.25, 0.01)

## plot graph
nx.draw_networkx(G)
plt.show()

## print numpy array of social graph
## ones means edge between nodes, otherwise zero
print(nx.to_numpy_matrix(G))
