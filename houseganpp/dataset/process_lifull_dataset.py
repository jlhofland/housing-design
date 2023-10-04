"""
This file will prepare the lifull dataset (housegan_clean_data.npy or train_data.npy/valid_data.npy) for training and validation in HeteroHouseGAN++

Inputs:
  
  A .npy file (housegan_clean_data.npy or train_data.npy/valid_data.npy) with the following information:
    *** House-GAN Dataset ***

    This dataset contains 145,811 floorplans in vector format utilized for training House-GAN. The data is organized in a list format, where each element represents one floorplan. For each floorplan in the dataset we have the following elements (in order):

    1) List of room types: mapping between room types and their corresponding class. ROOM_CLASS = {"exterior_wall": 0, "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}

    2) List of room bounding boxes: each bounding box is represented by [x0, y0, x1, y1], where (x0, y0) and (x1, y1) are bottom-left and top-right coordinates, respectively.

    3) List of floorplan edges: edges in the floorplan are represented by [x0, y0, x1, y1, *, *], where (x0, y0) and (x1, y1) are the edges endpoints and elements * are not being used (internally used by raster-to-vector).

    4) Edge to room mapping: for each edge we assign up to 2 rooms sharing that edge, normally we have 2 rooms for internal edges and 1 room for edges in the building footprint, cases with 0 room per edge (should not happen) are likely vectorization/processing errors.

    5) Doors to edges list: an element "i" in this list means that the i-th edge contains a door.

    6) Vector to RGB mapping: this field contains the name of the original RGB image from LIFULL dataset.

    
Outputs:
  
  A modified .npy file named "HHGPP_(train/eval)_data.npy" with the following information:
    A list of lists with entries defined below (length == number of valid LIFULL floorplans)
      [CHECK] "nds": all graph nodes with features in an (N x (11+2)) list. Each node is represented as a one-hot encoded vector with 11 classes, concatinated with the node features [length, door/no door], this is [-1,-1] for room nodes.
      [CHECK] "bbs": all graph node bounding boxes in an Nx4 list, including exterior wall nodes (need to expand the EW edges into single pixel-wide bbs, they have all been expanded 1 pixel in the positive x or y direction).
      [Done for RA edges, TODO for CE edges] "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
      [Done for RA edges, TODO for CE edges]  "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]

      Note that N == number of graph nodes, and E == number of graph edges

[ [ [nds],[bbs],[eds],[nds_f],[eds_f] ],
  [ [],[],[],[],[] ],
 ...]
"""

import torch
import numpy as np

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

original_data_path = 'houseganpp\dataset\housegan_clean_data.npy'
new_data_path = '../dataset/HHGPP_train_data.npy'

data = np.load(original_data_path, allow_pickle=True)


def is_adjacent(boxA, boxB, threshold = 0.03):
    xa0, ya0, xa1, ya1 = boxA
    xb0, yb0, xb1, yb1 = boxB

    ha, hb = xa1 - xa0, xb1 - xb0
    wa, wb = ya1 - ya0, yb1 - yb0

    xca, xcb = (xa0 + xa1) / 2.0, (xb0 + xb1) / 2.0
    yca, ycb = (ya0 + ya1) / 2.0, (yb0 + yb1) / 2.0
    
    delta_x = np.abs(xcb - xca) - (ha + hb) / 2.0
    delta_y = np.abs(ycb - yca) - (wa + wb) / 2.0

    delta = max(delta_x, delta_y)

    return delta < threshold

def is_edge_adjacent(edgeA, boxB, threshold = 0.03):
    xa0, ya0, xa1, ya1 = edgeA
    xb0, yb0, xb1, yb1 = boxB

    wa, wb = xa1 - xa0, xb1 - xb0
    ha, hb = ya1 - ya0, yb1 - yb0

    xca, xcb = (xa0 + xa1) / 2.0, (xb0 + xb1) / 2.0
    yca, ycb = (ya0 + ya1) / 2.0, (yb0 + yb1) / 2.0
    
    delta_x = np.abs(xcb - xca) - (wa + wb) / 2.0
    delta_y = np.abs(ycb - yca) - (ha + hb) / 2.0

    if ha == 0:
        return delta_x < threshold 

    if wa == 0:
        return delta_y < threshold

def get_exterior_walls(rooms_connected, edges, doors):
    indices = [i for i, item in enumerate(rooms_connected) if len(item) == 1]
    exterior_walls = [[i,edges[i],1] if i in doors else [i,edges[i],0] for i in indices]
    return exterior_walls

def edge_length(edge):
    x0, y0, x1, y1 = edge
    if x0 != x1 and y0 != y1:
        raise Exception("Edge length called for bounding box of width > 0")
    if x0 == x1:
        return abs(y1-y0)
    if y0 == y1:
        return abs(x1-x0)

def give_edge_width_1(edge):
    x0, y0, x1, y1 = edge
    if x0 != x1 and y0 != y1:
        raise Exception("Give edge width called for bounding box of width > 0")
    if x0 == x1:
        x1 = x1 + 1
    if y0 == y1:
        y1 = y1 + 1
    return [x0, y0, x1, y1]




new_data = []
for home in data:
    '''
      REMINDER Room nodes have no features [-1,-1]
      TODO Exterior walls as nodes with corresponding node features [length, door/no door], bounding boxes and edges
      TODO Edge features ([relative angle, door or not] for EW, [wall with door/wall without door/no wall, relative direction (E/NE/N/NW/W/SW/S/SE)])
      TODO Edges between exterior walls have to be added, I think they will follow from the algorithm used now, only the edge type will have to be changed.
    '''
    # Creating the node list (nodes) for later use and the list of one-hot encoded node vectors (nds), (Nx(11+2))
    nodes = home[0]

    nds = one_hot_embedding(nodes)
    nds = torch.FloatTensor(nds)
    nds = torch.cat((nds, torch.FloatTensor([[-1,-1] for i in range(len(nodes))])), 1)

    # Creating the bounding boxes for the rooms (Nx4)
    bbs = np.array([bb.tolist() for bb in home[1]]) / 256 # Values between 0 and 1



    # Find Exterior Walls and add to nodes and bounding boxes list
    exterior_walls = get_exterior_walls(home[3], np.array(home[2])[:,:4], home[4])
    ex_wall_nodes = []
    ex_wall_bbs = []
    for nod in exterior_walls:
        ex_wall_nodes.append([nod[0] + len(nds), edge_length(nod[1]), nod[2]])
        ex_wall_bbs.append(give_edge_width_1(nod[1]))
    ex_wall_nodes = np.array(ex_wall_nodes)
    ex_wall_bbs = np.array(ex_wall_bbs) / 256 # Bounding boxes of Exterior Walls values scaled between 0 and 1


    ew_nds = one_hot_embedding([0 for i in exterior_walls]) # Create nodes of type 0 for Exterior Walls
    ew_nds = torch.FloatTensor(ew_nds)
    ew_nds = torch.cat((ew_nds, torch.FloatTensor(ex_wall_nodes[:,1:])), 1) # Add EW node features to list

    nds = torch.cat((nds, ew_nds), 0) # Add room nodes and Exterior Wall nodes to the same nds node list
    
    bbs = np.concatenate((bbs, ex_wall_bbs), 0)

    # Creating the edges (Ex3), [src_node_id, +1/-1, dest_node_id]
    triples = []
    for k in range(len(nodes)):     # From each node
        for l in range(len(nodes)): # To each node
            if l > k:               # Each node pair only once (undirected)
                node0, bb0 = nodes[k], bbs[k]
                node1, bb1 = nodes[l], bbs[l]
                if is_adjacent(bb0, bb1):
                    triples.append([k,1,l])
                else:
                    triples.append([k,-1,l])
    triples = np.array(triples)
    eds = torch.LongTensor(triples)


    # Creating the edge features list (Ex3), [edge type (0 - CE, or 1 - RA), Door Feature, Angle Feature]
    # CREATE LIST WITH ALL EDGES WITH STANDARD FEATURES (DOOR FEATURE = 0 and Angle Feature = 0)
    all_edges = np.array(home[2])[:,:4] / 256 # all edges with values between 0 and 1

    edges_f = [[1, 0, 0] for i in range(len(eds))] # Create list with standard features [RA=1, Door=0, Dir=0]
    for id in home[4]: # For each edge in edges with doors list
        i=0
        for ed in range(len(eds)): # For each edge in eds
            if eds[ed][1] == 1: # But only if there is a room adjacency
                # print(f'checking adjacency with {all_edges[id]} for room {bbs[ed[0]]}')
                if is_edge_adjacent(all_edges[id], bbs[eds[ed][0]]): # Check whether edge with door is adjacent with both rooms of edge
                    if is_edge_adjacent(all_edges[id], bbs[eds[ed][2]]):
                        edges_f[i][1] = 1
    
    eds_f = torch.FloatTensor(edges_f)
            



    # # This converts bounding boxes into masks that fit the needed image size
    # im_size = 64
    # rooms_mks = np.zeros((len(nodes), im_size, im_size)) # Dim = (# nodes, image size, image size)
    # for k, (rm, bb) in enumerate(zip(nodes, bbs)):
    #   if rm > 0:
    #     x0, y0, x1, y1 = im_size*bb
    #     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    #     rooms_mks[k, x0:x1+1, y0:y1+1] = 1.0
    # rooms_mks = torch.FloatTensor(rooms_mks)
    
    
    

    # print('home = ')
    # print(home[0])
    # print('nodes = ')
    # print(nds)
    # print('bounding boxes = ')
    # print(bbs)
    # print('room masks = ')
    # print(rooms_mks)
    # print('triples = ')
    # print(triples)
    # print('edges = ')
    # print(eds)
    # print('edge features = ')
    # print(edges_f)
    # print('nodes features = ')
    # print(nds_f)

    break



# Finally, save the list:
# np.save(new_data_path, new_data)

