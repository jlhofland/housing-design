import dgl
import torch
import pickle

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
    return y[labels]

"""
Intent of this function is to convert a DGL-Heterograph into a list of nodes, edges, and edge features per the following format. 
This allows heterohouseganpp to then convert the graph data into a floor plan img.

    "nds": all graph nodes with features in an (Nx13) list. Each node is represented as a one-hot encoded vector with 11 classes, concatinated with the node features [length, door/no door], this is [-1,-1] for room nodes.
    "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
    "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]
for CE: edge feature 1 = angle between walls in degrees (-180 to 180) from EW_from to EW_to, edge feature 2 = 0
for RA: edge feature 1 = 0 (=wall with door) or 1 (=wall with no door), edge feature 2 = relative direction from room_from to room_to (0/1/2/3/4/5/6/7/8 = E/NE/N/NW/W/SW/S/SE/Undefined)

"""

graphlist_output_path = "housingpipeline/housingpipeline/floor_plan_pipeline/graphlist.pickle"

g = dgl.load_graphs("housingpipeline/housingpipeline/floor_plan_pipeline/misc/sample_graph.bin")[0][0]


hg = dgl.to_homogeneous(g, edata=['e'], store_type=True, return_count=True)



# # Uncomment to print out partial graph
# for c_et in g.canonical_etypes:
#     if g.num_edges(c_et) > 0:
#         print(f"Edge numbers: {c_et} : {g.num_edges(c_et)}")
#         print(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}")
# for nt in g.ntypes:
#     if g.num_nodes(nt) > 0:
#         print(f"Node features: {nt} :\n {g.nodes[nt].data}")



nds = []
eds = []
eds_f = []

names_node_types = g.ntypes # gives node names
ids_node_types = hg[0].ndata['_TYPE'] # gives list of the nodes represented by node type number
ew_features = g.nodes['exterior_wall'].data['hf'] # Gives edge features (list of 2 entries per edge)

node_type_counts = hg[1] # Number of nodes per type
num_nodes = torch.sum(torch.tensor(node_type_counts)).item() # Total number of nodes
canon_etype_counts = hg[2] # Number of edges per type
num_edges = torch.sum(torch.tensor(canon_etype_counts)).item() # Total number of edges

'''
Create node feature tensor
'''
# create empty node feature tensor
nds_f = torch.full((ids_node_types.shape[0],2), fill_value = -1.0)
# find nodetype id of EW's
ew_type_id = names_node_types.index('exterior_wall')
# find node ids of EW's
ew_ids = torch.argwhere(ids_node_types == ew_type_id).flatten()
# insert ew nd features
nds_f[ew_ids] = ew_features

'''
Create one-hot'd node type tensor
'''
nds_t = one_hot_embedding(labels=ids_node_types, num_classes=len(g.ntypes))

# Swap node type id numbers to what is used to train HHGPP: {"exterior_wall": 0, "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
nds_t[:,[0,1,2,3,4,5,6,7,8,9,10]] = nds_t[:,[6,9,7,2,1,10,3,0,4,5,8]] 

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



nodes_dict = {"exterior_wall": 0, "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
connection_dict = {"corner_edge": 0, "room_adjacency_edge": 1}


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



'''
Make list and save the graphlist to pickled file
'''

graphlist = [nds, eds, eds_f]

print(graphlist)

with open(graphlist_output_path, 'wb') as f:
  pickle.dump(graphlist,f)
  f.close()