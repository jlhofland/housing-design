import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch

# g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
# # Equivalently, PyTorch LongTensors also work.
# g = dgl.graph(
#     (torch.LongTensor([0, 0, 0, 0, 0]), torch.LongTensor([1, 2, 3, 4, 5])),
#     num_nodes=6,
# )

# # You can omit the number of nodes argument if you can tell the number of nodes from the edge list alone.
# g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]))

# # Assign a 3-dimensional node feature vector for each node.
# g.ndata["x"] = torch.randn(6, 3)
# # Assign a 4-dimensional edge feature vector for each edge.
# g.edata["a"] = torch.randn(5, 4)
# # Assign a 5x4 node feature matrix for each node.  Node and edge features in DGL can be multi-dimensional.
# g.ndata["y"] = torch.rande(6, 5, 4)

# print(g)

# Heterograph
# To set/get features for a specific node/edge type, DGL provides two new types of syntax:  
# g.nodes[‘node_type’].data[‘feat_name’] and g.edges[‘edge_type’].data[‘feat_name’].

graph_data = {
    # ('source node type', 'edge type', 'destination node type') : (torch.tensor([#A,#B]), torch.tensor([#C,#D]))
    # creates edges between node ID's A -> C, and B -> D
    # for multiple edges from the same node, must repeat its ID
    # Edges in DGLGraph are directed by default.
    # For undirected edges, add edges for both directions.
    ('bedroom', 'adjacent_to', 'exterior_wall') : (torch.tensor([0,0]), torch.tensor([0,1])),
    ('exterior_wall', 'adjacent_to', 'bedroom') : (torch.tensor([0,1]), torch.tensor([0,0])),
    ('exterior_wall', 'connects_to', 'exterior_wall') : (torch.tensor([0,1,2,3]), torch.tensor([1,2,3,0]))
}

g = dgl.heterograph(graph_data)

g.nodes['exterior_wall'].data['e'] = torch.tensor([1,99,12,123])

print(g)

print(g.nodes('bedroom'))

# print the 'e' feature value for exterior_wall node 1
print(g.nodes['exterior_wall'].data['e'][1])

# # Plot graph

# import matplotlib.pyplot as plt
# import networkx as nx

# options = {
#     'node_color': 'black',
#     'node_size': 20,
#     'width': 1,
# }
# G = dgl.to_networkx(g)
# plt.figure(figsize=[15,7])
# nx.draw(G, **options)
# plt.show()




# x = torch.randn(2, 3, 2)
# print(x)
# print(torch.cat((x, x, x), 2))


## JSON Input

import json
 
# Data to be written
user_layout_input = {
    "exterior_walls": [
        # Input exterior walls as a list of pairs of coordinates [ [ xi_0, yi_0, xf_0, yf_0 ], [ xi_1, yi_1, xf_1, yf_1 ] , … ]. 
        # Ensure that exterior walls form a chain. Thus, xf_0 == xi_1 and xf_N == xi_0, and for y. 
        [0,0,0,1],
        [0,1,1,1],
        [1,1,1,0],
        [1,0,0,0]
    ],
    "number_of_living_rooms": 1,
    "living_rooms_plus?" : False,
    "number_of_bedrooms": 1,
    "bedrooms_plus?" : False,
    "number_of_bathrooms": 1,
    "bathrooms_plus?" : False,
    "connections": [
        # [room_type, room_index, room_type, room_index, add_door?]
        # Room types: 0 == exterior_wall, 1 == door, 2 == living_room, 3 == bedroom, 4 == bathroom 
        [0, 3, 1, 0],
        [2, 0, 1, 0],
        [2, 0, 1, 1],
        [2, 0, 1, 2],
        [3, 0, 1, 1],
        [4, 0, 1, 2],
        [3, 0, 0, 0],
        [4, 0, 0, 2]
    ]
}
 
# # Serializing json
# json_object = json.dumps(user_layout_input, indent=4)
 
# # Writing to sample.json
# with open("sample.jsonc", "w") as outfile:
#     outfile.write(json_object)


# Reading from sample.json
# Opening JSON file
with open('sample.jsonc', 'r') as openfile:
 
    # Reading from json file into python dict
    layout = json.load(openfile)
    a = json.load()
 
print(layout)
print(type(layout))
print(layout['connections'])
