import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

""" args
groups: list of groups
inner: percentage of edges within groups
outer: percentage of edges between groups
"""

def social_graph(groups, inner, outer, plot):
    # create graph
    G = nx.random_partition_graph(groups, inner, outer)

    # plot graph
    if plot == True:
        nx.draw_networkx(G)
        plt.show()

    # print numpy array of social graph
    # ones means edge between nodes, otherwise zero
    c = nx.to_numpy_matrix(G)
    return c
