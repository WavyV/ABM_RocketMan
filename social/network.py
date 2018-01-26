import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
import collections

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
    return c, G

# random graph: Barabási–Albert model
def barabasi_albert(n, m, plot = False):
    G = nx.barabasi_albert_graph(n,m)

    # plot graph
    if plot == True:
        nx.draw_networkx(G)
        plt.show()

    # return a numpy matrix (undirected)
    c = nx.to_numpy_matrix(G)
    return c, G

def graph_to_histogram(G):
    degree_sequence = sorted([d for n, d in G.degree().items()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

if __name__ == "__main__":
    # c = erdos_renyi(75, 16.68/75, True)
    # print(c)
    n, m = 75, 7
    c, G = barabasi_albert(n, m, plot = False)
    graph_to_histogram(G)
