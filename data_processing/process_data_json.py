import json
import networkx as nx

# load json data
with open('test-form.json') as json_data:
    data = json.load(json_data)

# create variable to store data
list_degree = []

# iterate over json data and append degrees
for i in data:
    if int(i['coursefriends']) > len(data):
        list_degree.append(len(data))
    else:
        list_degree.append(int(i['coursefriends']))

# create configuration model (aka social network)
G = nx.random_degree_sequence_graph(list_degree)

# plot graph
nx.draw_networkx(G)
plt.show()
