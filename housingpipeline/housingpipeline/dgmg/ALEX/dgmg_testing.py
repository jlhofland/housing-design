import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch

g : dgl.DGLGraph = dgl.load_graphs("housingpipeline/housingpipeline/dgmg/example_graphs/dgmg_graph_0.bin")[0][0]

hg = dgl.to_homogeneous(g, edata=['e'], store_type=True, return_count=True)

nodes_dict = {"exterior_wall": 0, "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
connection_dict = {"corner_edge": 0, "room_adjacency_edge": 1}



def one_hot_embedding(labels, num_classes=11):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    # print(" label is",labels)
    return y[labels.int()]


# src_types = dict()
# src_etypes = dict()
# for cet in g.canonical_etypes:
#     src = cet[0]
#     if src == "exterior_wall": continue
#     if g.num_edges(cet) == 0: continue
#     print(f"Num {cet} = {g.num_edges(cet)}")
#     if src not in src_types.keys():
#         src_types[src] = g.num_nodes(src)
#     if src not in src_etypes.keys():
#         src_etypes[src] = set()
#     src_etypes[src].add(cet)

# print(f"Source types: {src_types}")
# for src in src_etypes.keys():
#     print(f"{src}:")
#     for cet in src_etypes[src]:
#         print(cet)
#     # print(src_etypes[src])

# print("\n")
# for src in src_types.keys():
#     if src != 'corridor': continue
#     for src_id in range(src_types[src]):
#         if src_id > 0: continue
#         print(f"Source {src} ID {src_id}")
#         for cet in src_etypes[src]:
#             src_out_edges = g.out_edges(src_id, etype=cet)
#             if src_out_edges[0].shape[0] != 0:
#                 print(cet)
#                 print(src_out_edges)



# Transform dgl graph into graphlist format
nds = []
eds = []
eds_f = []

ew_features = g.nodes['exterior_wall'].data['hf'] # Gives edge features (list of 2 entries per edge)

ids_nodes = []

for key in g.ndata.get('a').keys():
    keyamount = len(g.ndata.get('a').get(key))
    while keyamount > 0:
        ids_nodes.append(key)
        keyamount += -1

ids_node_types = []

for item in ids_nodes:
    ids_node_types.append(int(nodes_dict.get(item)))
ids_node_types = torch.Tensor(ids_node_types)

#node_type_counts = hg[1] # Number of nodes per type
#num_nodes = torch.sum(torch.tensor(node_type_counts)).item() # Total number of nodes
#canon_etype_counts = hg[2] # Number of edges per type
#num_edges = torch.sum(torch.tensor(canon_etype_counts)).item() # Total number of edges

'''
Create node feature tensor
'''
# create empty node feature tensor
nds_f = torch.full((ids_node_types.shape[0],2), fill_value = -1.0)
# find nodetype id of EW's
ew_type_id = nodes_dict.get('exterior_wall')
# find node ids of EW's
ew_ids = torch.argwhere(ids_node_types == ew_type_id).flatten()
# insert ew nd features
nds_f[ew_ids] = ew_features
'''
Create one-hot'd node type tensor
'''
nds_t = one_hot_embedding(labels=ids_node_types, num_classes=len(g.ntypes))

# Swap node type id numbers to what is used to train HHGPP: {"exterior_wall": 0, "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
# nds_t[:,[0,1,2,3,4,5,6,7,8,9,10]] = nds_t[:,[6,9,7,2,1,10,3,0,4,5,8]]

'''
Create final nds list
'''
nds = torch.concatenate([nds_t, nds_f], dim=1)




'''
Create initial edge tensor with all edges (from_id, -1, to_id)
'''

# Create eds list of all connections with -1 (no edge)
# and create empty/default eds_f list with in and out node id entries will be omitted later, but for used for filling the list)
eds = []
for i, node_i in enumerate(ids_node_types):
    for j, node_j in enumerate(ids_node_types):
        if i != j:
            eds.append([i, -1, j])

# Create empty (or default) eds_f list
eds_f = [[0,0,0] for i in range(len(eds))]

# Create list that shows what the id of the first node is for each node type
first_id_of_room_type = [0 for i in range(11)]
for room_type in range(11):
    room_exists = 0
    for i, nodevector in enumerate(nds):
        if nodevector[room_type] == 1:
            first_id_of_room_type[room_type] = i
            room_exists = 1
        if room_exists == 1:
            break



'''
Fill the eds and eds_f lists
'''

for etype in g.canonical_etypes:
    from_node, connection_type, to_node = etype
    from_node = nodes_dict.get(from_node)
    connection_type = connection_dict.get(connection_type)
    to_node = nodes_dict.get(to_node)

    from_ids, to_ids = g.edges(etype = etype)
    
    if len(from_ids) != 0:
        for i,j in enumerate(from_ids):
            from_id = first_id_of_room_type[from_node] + from_ids[i]
            to_id = first_id_of_room_type[to_node] + to_ids[i]
            edge_feature1 = g.edata.get('e').get(etype)[i][0].item()
            edge_feature2 = g.edata.get('e').get(etype)[i][1].item()

            # Add edge and edge features
            for ed_number,ed in enumerate(eds):
                if from_id == ed[0] and to_id == ed[2]:
                    eds[ed_number][1] = 1
                    eds_f[ed_number][0] = connection_type
                    eds_f[ed_number][1] = edge_feature1
                    eds_f[ed_number][2] = edge_feature2


bbs = []
graphlist = [bbs, nds, eds, eds_f]


# Now check for the direction distribution
num_directions_per_room = []
for i, room in enumerate(graphlist[1]): # Check each room
    room_direction_list = []
    num_directions = 0
    North = 0
    East = 0
    South = 0
    West = 0

    if room[0] == 0.: # Not EW
        for j,edge in enumerate(graphlist[2]): # Check each edge
            if edge[1] == 1: # If edge exists
                if edge[0] == i: # If edge is from given room
                    room_direction_list.append(graphlist[3][j][2])
                    # print(room_direction_list)
        # print(f'direction going out of room {i}')
        # print(room_direction_list)
        # print('')
        
        for direction in room_direction_list:
            if direction in {7,0,1}: # direction is SE, E, NE
                East = 1
            if direction in {1,2,3}: # direction is NE, N, NW
                North = 1
            if direction in {3,4,5}: # direction is NW, W, SW
                West = 1
            if direction in {5,6,7}: # direction is SW, S, SE
                South = 1
        num_directions = East + North + West + South
        # print(num_directions)
        # direction_list.append(room_direction_list)
        num_directions_per_room.append(num_directions)


room_distribution = []
for i in range(5):
    room_distribution.append(num_directions_per_room.count(i) / len(num_directions_per_room))

print(f'Distribution of amount of rooms ([0,1,2,3,4] rooms): {room_distribution}')

# print('\n')
# print(g.edata)

room_without_door_counter = 0
for i, room in enumerate(graphlist[1]): # Check each room
    room_has_door = False
    if room[0] == 0.:
        for edg in graphlist[2]:
            if edg[0] == i: # If edge is from room
                if graphlist[3][i][1] == 0: # If edge has door
                    room_has_door = True
                if graphlist[1][edg[2]][0] == 0.: # To node is exterior wall
                    if graphlist[1][edg[2]][12] == 1:
                        room_has_door = True
        if not room_has_door:
            room_without_door_counter += 1
            # print(f'{i}: {room}')

print(f'Number of rooms with no door: {room_without_door_counter}')

# for jj,eee in enumerate(graphlist[2]):
#     print(f'{eee} + {graphlist[3][jj]}')
# print(g.edata)
# print(g.ndata)




###########################################################################################

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


# ###########################################################################################

# # New heterograph - try to define all node and edge types up-front...
# room_type_mapping = {0:"exterior_wall", 1:"living_room", 2:"kitchen", 3:"bedroom", 4:"bathroom", 5:"missing", 6:"closet", 7:"balcony", 8:"corridor", 9:"dining_room", 10:"laundry_room"}  

# graph_data = {
#     ('bedroom', 'room_adjacency', 'exterior_wall') : (torch.tensor([0]), torch.tensor([0])),
#     ('bathroom', 'room_adjacency', 'exterior_wall') : (torch.tensor([0]), torch.tensor([0])),
# }

# g = dgl.heterograph(graph_data)



# def define_empty_typed_graph(ntypes, etypes):

#     def remove_all_edges(g):
#         for etype in g.canonical_etypes:
#             num_eids = g.num_edges(etype)
#             eids = list(range(num_eids))
#             g.remove_edges(eids=eids, etype=etype)

#     def remove_all_nodes(g):
#         for ntype in g.ntypes:
#             num_nids = g.num_nodes(ntype)
#             nids = list(range(num_nids))
#             g.remove_nodes(nids=nids, ntype=ntype)

#     def empty_out_graph(g):
#         remove_all_edges(g)
#         remove_all_nodes(g)

#     graph_data = {}
#     for etype in etypes:
#         for src_ntype in ntypes:
#             # print(f"src: {src_ntype}")
#             # print(f"ntypes: {ntypes}")
#             dest_ntypes = ntypes.copy()
#             dest_ntypes.remove(str(src_ntype))
#             # print(f"dest_ntypes: {dest_ntypes}")
#             for dest_ntype in dest_ntypes:
#                 canonical_etype = (src_ntype, etype, dest_ntype)
#                 nids = (torch.tensor([0]), torch.tensor([0]))
#                 graph_data[canonical_etype] = nids

#     g = dgl.heterograph(graph_data)
#     empty_out_graph(g)
#     return g

# room_types = ["exterior_wall", "living_room"]#, "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room"] 
# edges_types = ["connection_corner", "room_adjacency"]
# g = define_empty_typed_graph(room_types, edges_types)
# g.add_edges(u=0, v=0, data={'hf': torch.tensor([[2,2]])}, etype=('living_room', 'room_adjacency', 'exterior_wall'))
# g.add_edges(u=0, v=1, etype=('living_room', 'room_adjacency', 'exterior_wall'))

# # try to add a feature vector
# # g.nodes['exterior_wall'].data['hf'] = torch.tensor([2])

# g.add_edges(u=0, v=2, etype=('living_room', 'room_adjacency', 'exterior_wall'))
# for c_et in g.canonical_etypes:
#     if g.num_edges(c_et) > 0:
#         print(f"Edge numbers: {c_et} : {g.num_edges(c_et)}")
#         # print(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}")
# print(g.ndata)
# print(g.num_nodes('exterior_wall'))
# # g.nodes['exterior_wall'].data['hf'] = torch.tensor([2]).repeat(11,1)
# print(g.edata['hf'])

###########################################################################################

# # Heterograph
# # To set/get features for a specific node/edge type, DGL provides two new types of syntax:  
# # g.nodes[‘node_type’].data[‘feat_name’] and g.edges[‘edge_type’].data[‘feat_name’].

# graph_data = {
#     # ('source node type', 'edge type', 'destination node type') : (torch.tensor([#A,#B]), torch.tensor([#C,#D]))
#     # creates edges between node ID's A -> C, and B -> D
#     # for multiple edges from the same node, must repeat its ID
#     # Edges in DGLGraph are directed by default.
#     # For undirected edges, add edges for both directions.
#     ('bedroom', 'adjacent_to', 'exterior_wall') : (torch.tensor([0,0]), torch.tensor([0,1])),
#     ('exterior_wall', 'adjacent_to', 'bedroom') : (torch.tensor([0,1]), torch.tensor([0,0])),
# }

# print("Add new nodes/edges")
# graph_data[('exterior_wall', 'connects_to', 'exterior_wall')] = (torch.tensor([0,1,2,3]), torch.tensor([1,2,3,0]))

# print(f"GRAPH DATA:\n {graph_data}")
# print(f"Extracting?\n {graph_data[('bedroom', 'adjacent_to', 'exterior_wall')]}")

# g = dgl.heterograph(graph_data)

# g.nodes['bedroom'].data['e'] = torch.tensor([1,99,12,123]).repeat(g.num_nodes('bedroom'),1)
# g.nodes['exterior_wall'].data['e'] = torch.rand(g.num_nodes('exterior_wall'), 1, 7)


# print(f"Number of nodes: {g.num_nodes()}")
# print(f"Number of node types: {len(g.ntypes)}")
# print(f"Graph node types: {g.ntypes}")
# print(f"Graph edge types: {g.etypes}")
# print(f"Number of nodes of type 'bedroom': {g.num_nodes('bedroom')}")
# print(f"Number of nodes of type 'exterior_wall': {g.num_nodes('exterior_wall')}")
# print(f"Node id's for nodes of type 'bedroom': {g.nodes('bedroom')}") # Note the () here
# print(f"Node id's for nodes of type 'exterior_wall': {g.nodes('exterior_wall')}")
# print(f"Node data for nodes of type 'bedroom':\n {g.nodes['bedroom']}") # and the [] here!
# print(f"Node data for nodes of type 'exterior_wall':\n {g.nodes['exterior_wall']}")
# print(f"'e' feature value for exterior_wall node 1: {g.nodes['exterior_wall'].data['e'][1]}")
# print("Changing the 'e' data for exterior_wall node 1")
# g.nodes['exterior_wall'].data['e'][1] = torch.tensor([[1,1,1,37,1,1,1]])
# print(f"'e' feature value for exterior_wall node 1: {g.nodes['exterior_wall'].data['e'][1]}")

# print("CONVERTING TO HOMOGENEOUS GRAPH")
# homo_g = dgl.to_homogeneous(g)
# print(homo_g.ndata)


###########################################################################################


# print("\n\n New Graph: \n\n")
# g2 = dgl.graph(([0, 0], [1, 2]))
# g2.ndata['_TYPE'] = torch.tensor([0, 1, 2])
# g2.ndata['_ID'] = torch.tensor([0, 0, 0])
# g2.edata['_TYPE'] = torch.tensor([0, 1])
# g2.edata['_ID'] = torch.tensor([0, 0])
# print(f"Homogeneous graph with n- and e-data added:\n {g2}")
# ndata = ['base', 'type1', 'type2']
# edata = ['connects', 'fucks']
# hg = dgl.to_heterogeneous(g2, ndata, edata)
# print(f"Heterograph made from homo graph:\n {hg}")


# print(f"Merging the two hetero-graphs? No...\n")
# # G = dgl.merge([g, hg])
# # print(G)

# print(f"Can we add in the first graph mannually? Maybe ...")

# # hg.add_edges()

###########################################################################################
# Plotting


# graph_data = {
#     ('bedroom', 'room_adjacency', 'exterior_wall') : (torch.tensor([0,1]), torch.tensor([0,0])),
#     ('bathroom', 'room_adjacency', 'exterior_wall') : (torch.tensor([0]), torch.tensor([0])),
# }

# g = dgl.heterograph(graph_data)

# # Get node-type order
# node_type_order = g.ntypes

# # Create node-type subgraph
# g_homo = dgl.to_homogeneous(g)
# print(g_homo.ndata)

# labels = {}
# for idx, node in enumerate(g_homo.ndata[dgl.NTYPE]):
#     labels[idx] = node_type_order[node] + "_" + str(int(g_homo.ndata[dgl.NID][idx]))
# print(labels)

# # Plot graph

# import matplotlib.pyplot as plt
# import networkx as nx
# colors = ['pink', 'purple', 'lightblue']
# labels = {0: 'bathroom_0', 1: 'bedroom_0', 2: 'exterior_wall_0'}
# options = {
#     # 'node_color': colors,
#     'node_size': 300,
#     'width': 1,
#     # 'labels': labels,
#     'font_size': 20
# }
# G = dgl.to_networkx(g_homo)
# plt.figure(figsize=[15,7])
# nx.draw(G, with_labels = True, labels = labels, node_color=colors, font_color='r', **options)
# plt.show()




# x = torch.randn(2, 3, 2)
# print(x)
# print(torch.cat((x, x, x), 2))


# ## JSON Input

# import json
 
# # Data to be written
# user_layout_input = {
#     "exterior_walls": [
#         # Input exterior walls as a list of pairs of coordinates [ [ xi_0, yi_0, xf_0, yf_0 ], [ xi_1, yi_1, xf_1, yf_1 ] , … ]. 
#         # Ensure that exterior walls form a chain. Thus, xf_0 == xi_1 and xf_N == xi_0, and for y. 
#         [0,0,0,1],
#         [0,1,1,1],
#         [1,1,1,0],
#         [1,0,0,0]
#     ],
#     "number_of_living_rooms": 1,
#     "living_rooms_plus?" : False,
#     "number_of_bedrooms": 1,
#     "bedrooms_plus?" : False,
#     "number_of_bathrooms": 1,
#     "bathrooms_plus?" : False,
#     "connections": [
#         # [room_type, room_index, room_type, room_index, add_door?]
#         # Room types: 0 == exterior_wall, 1 == door, 2 == living_room, 3 == bedroom, 4 == bathroom 
#         [0, 3, 1, 0],
#         [2, 0, 1, 0],
#         [2, 0, 1, 1],
#         [2, 0, 1, 2],
#         [3, 0, 1, 1],
#         [4, 0, 1, 2],
#         [3, 0, 0, 0],
#         [4, 0, 0, 2]
#     ]
# }
 
# # # Serializing json
# # json_object = json.dumps(user_layout_input, indent=4)
 
# # # Writing to sample.json
# # with open("sample.jsonc", "w") as outfile:
# #     outfile.write(json_object)


# # Reading from sample.json
# # Opening JSON file
# with open('sample.jsonc', 'r') as openfile:
 
#     # Reading from json file into python dict
#     layout = json.load(openfile)
#     a = json.load()
 
# print(layout)
# print(type(layout))
# print(layout['connections'])







# x = torch.randn(2, 3, 2)
# print(x)
# print(torch.cat((x, x, x), 2))


## JSON Input

# import json
 
# # Data to be written
# user_layout_input = {
#     "exterior_walls": [
#         # Input exterior walls as a list of pairs of coordinates [ [ xi_0, yi_0, xf_0, yf_0 ], [ xi_1, yi_1, xf_1, yf_1 ] , … ]. 
#         # Ensure that exterior walls form a chain. Thus, xf_0 == xi_1 and xf_N == xi_0, and for y. 
#         [0,0,0,1],
#         [0,1,1,1],
#         [1,1,1,0],
#         [1,0,0,0]
#     ],
#     "number_of_living_rooms": 1,
#     "living_rooms_plus?" : False,
#     "number_of_bedrooms": 1,
#     "bedrooms_plus?" : False,
#     "number_of_bathrooms": 1,
#     "bathrooms_plus?" : False,
#     "connections": [
#         # [room_type, room_index, room_type, room_index, add_door?]
#         # Room types: 0 == exterior_wall, 1 == door, 2 == living_room, 3 == bedroom, 4 == bathroom 
#         [0, 3, 1, 0],
#         [2, 0, 1, 0],
#         [2, 0, 1, 1],
#         [2, 0, 1, 2],
#         [3, 0, 1, 1],
#         [4, 0, 1, 2],
#         [3, 0, 0, 0],
#         [4, 0, 0, 2]
#     ]
# }
 
# # Serializing json
# json_object = json.dumps(user_layout_input, indent=4)
 
# # Writing to sample.json
# with open("sample.jsonc", "w") as outfile:
#     outfile.write(json_object)


# # Reading from sample.json
# # Opening JSON file
# with open('sample.jsonc', 'r') as openfile:
 
#     # Reading from json file into python dict
#     layout = json.load(openfile)
#     a = json.load()
 
# print(layout)
# print(type(layout))
# print(layout['connections'])