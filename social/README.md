# Various random graph generators

In network.py, there are various functions to generate random social graphs. Unfortunately it is not possible to recreate an exact social network from the questionaires.

## erdos_renyi(n, p, plot = False)

Returns a random graph, also known as an Erdős-Rényi graph or a binomial graph.

The model chooses each of the possible edges with probability p.

Following the data from Friday January 19th, 75 people gave information about number of friends and their average number of friends were 16.68. This results in a probability p of 16.68 / 75 = 0.22. The algorithm generates a random graph with n (= 75) and p (= 0.22) and translates it into a numpy array.

## barabasi_albert(n, m, plot = False)

Returns a random graph according to the Barabási–Albert preferential attachment model.

A graph of n
nodes is grown by attaching new nodes each with m edges that are preferentially attached to existing nodes with high degree.

Still figuring out how to use this algorithm properly, however m = 3 gives a nice graph
