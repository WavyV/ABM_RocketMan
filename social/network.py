import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy

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

# random graph: Erdős - Renyi model
def erdos_renyi(n, p, plot = False):
    # create graph
    G = nx.erdos_renyi_graph(n,p)

    # plot graph
    if plot == True:
        nx.draw_networkx(G)
        plt.show()

    # return a numpy matrix (undirected)
    c = nx.to_numpy_matrix(G)
    return c

# random graph: Barabási–Albert model
def barabasi_albert(n, m, plot = False):
    G = nx.barabasi_albert_graph(n,m)

    # plot graph
    if plot == True:
        nx.draw_networkx(G)
        plt.show()

    # return a numpy matrix (undirected)
    c = nx.to_numpy_matrix(G)
    return c

if __name__ == "__main__":
    # c = erdos_renyi(75, 16.68/75, True)
    # print(c)
    c = barabasi_albert(75, 3, plot = True)
