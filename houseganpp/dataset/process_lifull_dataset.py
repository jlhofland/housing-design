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
      [CHECK] "nds": all graph nodes with features in an (Nx13) list. Each node is represented as a one-hot encoded vector with 11 classes, concatinated with the node features [length, door/no door], this is [-1,-1] for room nodes.
      [CHECK] "bbs": all graph node bounding boxes in an Nx4 list, including exterior wall nodes (need to expand the EW edges into single pixel-wide bbs, they have all been expanded 1 pixel in the positive x or y direction).
      [CHECK] "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
      [TODO]  "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]

      Note that N == number of graph nodes, and E == number of graph edges

[ [ [nds],[bbs],[eds],[eds_f] ],
  [ [],[],[],[],[] ],
 ...]
"""

import torch
import numpy as np
import random

original_data_path = 'houseganpp\dataset\housegan_clean_data.npy'
new_data_path = 'houseganpp/dataset/HHGPP_train_data.npy'

data = np.load(original_data_path, allow_pickle=True)



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

def edge_has_door(bb1, bb2, edges, doorlist):
    # returns whether there is a door between bb1 and bb2 given list of edges and list of edges with doors
    return True

def relative_direction(bb1, bb2):
    # Gives relative direction from bb1 to bb2
    angle = 1
    direction = 1
    return direction

def find_angle_EW(bb1, bb2):
    # Gives angle between EW1 and EW2
    return 9






new_data = []

abc=0 # used for restricting amount of homes processed

for home in data:
    
    '''
      REMINDER Room nodes have no features [-1,-1]
      TODO Edge features ([relative angle, 0] for EW, for RA [(wall/wall with door) = (0/1), relative direction (E/NE/N/NW/W/SW/S/SE)=(0/1/2/3/4/5/6/7)])
      TODO Change edge types of edges between rooms
    '''

    # Creating the node list (nodes) for later use and the list of one-hot encoded node vectors (nds), (Nx(11+2))
    rooms = home[0]

    nds = one_hot_embedding(rooms)
    nds = torch.FloatTensor(nds)
    nds = torch.cat((nds, torch.FloatTensor([[-1,-1] for i in range(len(rooms))])), 1) # Add features [-1,-1] to room nodes

    # Creating the bounding boxes for the rooms (Nx4)
    bbs = np.array([bb.tolist() for bb in home[1]]) / 256 # Values between 0 and 1


    # Find Exterior Walls and add to nodes (with features length and door) and bounding boxes list
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
    

    bbs = np.concatenate((bbs, ex_wall_bbs), 0) # Add bounding boxes of EW to bounding boxes list


    # Creating the edges (Ex3), [src_node_id, +1/-1, dest_node_id]
    triples = []
    for k in range(len(nds)):     # From each node
        for l in range(len(nds)): # To each node
            if l > k:               # Each node pair only once (undirected)
                node0, bb0 = nds[k], bbs[k]
                node1, bb1 = nds[l], bbs[l]
                if is_adjacent(bb0, bb1):
                    triples.append([k,1,l])
                else:
                    triples.append([k,-1,l])
    triples = np.array(triples)
    eds = triples


    # Creating the edge features list (Ex3), [edge type (0 - CE, or 1 - RA), Door Feature, Angle Feature]
    # CREATE LIST WITH ALL EDGES WITH STANDARD FEATURES (DOOR FEATURE = 0 and Angle Feature = 0)
    all_edges = np.array(home[2])[:,:4] / 256 # all edges with values between 0 and 1


    edges_f = [[0, 0, 0] for i in range(len(eds))] # Create list for features [0, 0, 0]

    '''
    For each edge [from, connected, to], if from or to is a room: edge type is RA (1) with door feature (0/1) and relative direction (E/NE/N/NW/W/SW/S/SE)=(0/1/2/3/4/5/6/7)
                                    if from and to are Exterior Wall: edge type is CE (0) with angle feature in degrees and 0.
    '''
    i=0
    for edge in eds:
        if edge[0] in range(len(rooms)):  # If 'from' node is a room
            edges_f[i][0] = 1 # Edge type is RA
            # edges_f[i][2] = relative_direction(bbs[edge[0]], bbs[edge[2]]) # Find relative direction
            edges_f[i][2] = random.randint(0,7)

            if edge[2] in range(len(rooms)): # If 'to' node is a room then check for door
                #if edge_has_door(bbs[edge[0]], bbs[edge[2]], home[3], home[5]):
                #    edges_f[i][1] = 1

                if i > 2: # Random door between rooms, but make sure there is at least two doors
                    edges_f[i][1] = random.randint(0,1)
                else:
                    edges_f[i][1] = 1 

        # The order of the nodes is always first the rooms and then the Exterior Walls, so if edge[0] is an EW, then so is edge[2]. MIGHT NEED TO BE CHANGED IF WE WANT DOUBLE (DIRECTED) EDGES
        else: # Connection between two EW
            # edges_f[i][1] =  find_angle_EW(bbs[edge[0]], bbs[edge[2]])# Find relative angle between EW and add to 2nd feature, all other features ar 0.
            edges_f[i][1] = random.randint(-180,180)
            
        i = i + 1  

    edges_f = np.array(edges_f)


    # for id in home[4]: # For each edge in edges with doors list from LIFULL
    #     i=0
    #     for ed in range(len(eds)): # For each edge in eds (all possible connections between nds)
    #         if eds[ed][1] == 1: # But only if there is a room adjacency
    #             if is_edge_adjacent(all_edges[id], bbs[eds[ed][0]]): # Check whether edge with door is adjacent with both rooms of edge
    #                 if is_edge_adjacent(all_edges[id], bbs[eds[ed][2]]):
    #                     edges_f[i][1] = 1
    #         i = i+1
    # eds_f = torch.FloatTensor(edges_f)
            



    # # This converts bounding boxes into masks that fit the needed image size
    # im_size = 64
    # rooms_mks = np.zeros((len(nds), im_size, im_size)) # Dim = (# nodes, image size, image size)
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
    # print('edges = ')
    # print(eds)
    # print('edge features = ')
    # print(edges_f)
    new_data.append([nds, bbs, eds, edges_f])
    

    abc = abc + 1
    if abc == 3: # amount of rooms we want in the list
        break



print(new_data)

# Finally, save the list:

# np.save(new_data_path, new_data)
########################################
# Gives an error because the lists nds, bbs, eds and edges_f are not the same shape. Can not be put into an np.array. Perhaps try in CSV format?...
########################################