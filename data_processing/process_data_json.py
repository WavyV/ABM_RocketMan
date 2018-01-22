import json
import os
import networkx as nx
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../data")

# load json data
with open(os.path.join(data_path, 'fri-form.json')) as json_data:
    data = json.load(json_data)

# create variable to store data
list_degree = []

# iterate over json data and append degrees
for i in data:
    try:
        if int(i['coursefriends']) > len(data):
            list_degree.append(len(data))
        else:
            list_degree.append(int(i['coursefriends']))
    except:
        pass

print(list_degree)
print("length of list = " + str(len(list_degree)))
print(np.mean(np.asarray(list_degree)))
