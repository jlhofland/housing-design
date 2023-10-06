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
  
  A modified .npy file named "HHGPP_(train/eval/test)_data.npy" with the following information:
    A list of lists with entries defined below (length == number of valid LIFULL floorplans)
      [CHECK] "nds": all graph nodes with features in an (Nx13) list. Each node is represented as a one-hot encoded vector with 11 classes, concatinated with the node features [length, door/no door], this is [-1,-1] for room nodes.
      [CHECK] "bbs": all graph node bounding boxes in an Nx4 list, including exterior wall nodes (EW have all been expanded 1 pixel in the positive x or y direction).
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
import matplotlib.pyplot as plt

original_data_path = 'houseganpp/dataset/housegan_clean_data.npy'
new_data_path = 'houseganpp/dataset/HHGPP_train_data.npy'

data = np.load(original_data_path, allow_pickle=True)

ROOM_CLASS = {0: "exterior_wall", 1:"living_room", 2:"kitchen", 3:"bedroom", 4:"bathroom", 5:"missing", 6:"closet", 7:"balcony", 8:"corridor", 9:"dining_room", 10:"laundry_room"}

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

# def is_edge_adjacent(edgeA, boxB, threshold = 0.01):
#     # This threshold allows for 2 px of error (for 256x256 image)
#     # Returns whether edgeA and boxB are adjacent. Perpendicular also returns true.
#     xa0, ya0, xa1, ya1 = edgeA
#     xb0, yb0, xb1, yb1 = boxB

#     wa, wb = xa1 - xa0, xb1 - xb0
#     ha, hb = ya1 - ya0, yb1 - yb0

#     xca, xcb = (xa0 + xa1) / 2.0, (xb0 + xb1) / 2.0
#     yca, ycb = (ya0 + ya1) / 2.0, (yb0 + yb1) / 2.0
    
#     delta_x = np.abs(xcb - xca) - (wa + wb) / 2.0
#     delta_y = np.abs(ycb - yca) - (ha + hb) / 2.0

#     delta = max(delta_x, delta_y)

#     return delta < threshold

def room_edge_adjacent(room, edge, edges_room_mapping, bbs):
    ''' Checks if room is adjacent to edge '''

    x0, y0, x1, y1 = np.array(bbs[edge]) * 256
    if x1 == x0 + 1:
        x1 = x0
    if y1 == y0 + 1:
        y1 = y0
    edge_bb = [x0, y0, x1, y1]
    for i in range(len(home[2])):
        bb = home[2][i][:4]
        if edge_bb == bb:
            if room in edges_room_mapping[i]:
                return True
    return False

def rooms_adjacent(room1, room2, edges_room_mapping):
    ''' Returns whether the 2 rooms are adjacent by checking the edges to rooms mappings '''
    for edge_rooms in edges_room_mapping:
        if room1 in edge_rooms and room2 in edge_rooms:
            return True
    return False


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

def edge_has_door(room1, room2, edges_room_mapping, doorlist):
    ''' returns whether there is a door between room1 and roo2 given list of edges_to_room_mappings and list of edges with doors '''
    
    # Make list with all edges that have doors in the home
    edges_with_doors_room_mapping = []
    for d in doorlist:
        edges_with_doors_room_mapping.append(edges_room_mapping[d])
    for edge_rooms in edges_with_doors_room_mapping:
        if room1 in edge_rooms and room2 in edge_rooms:
            return True
    return False

def relative_direction(bb1, bb2):
    # Gives relative direction from bb1 to bb2
    # (E/NE/N/NW/W/SW/S/SE)=(0/1/2/3/4/5/6/7)
    x0, y0, x1, y1 = bb1
    x2, y2, x3, y3 = bb2
    c1 = [(x0 + x1) / 2, (y0 + y1) / 2]
    c2 = [(x2 + x3) / 2, (y2 + y3) / 2]
    vector = np.array([c2[0] - c1[0], c2[1] - c1[1]])
    direction = 8 # Undefined
    angle = np.degrees(np.arctan2(vector[1],vector[0]))
    if -22.5 < angle <= 22.5:
        direction = 0
    if 22.5 < angle <= 67.5:
        direction = 1
    if 67.5 < angle <= 112.5:
        direction = 2
    if 112.5 < angle <= 157.5:
        direction = 3
    if 157.5 < angle <= 180 or -180 < angle <= -157.5:
        direction = 4
    if -157.5 < angle <= -112.5:
        direction = 5
    if -112.5 < angle <= -67.5:
        direction = 6
    if -67.5 < angle <= -22.5:
        direction = 7
        
    return direction

def find_angle_EW(bbA, bbB):
    ''' TODO DOES NOT RETURN CORRECT ANGLE '''
    # Gives angle between EW1 and EW2
    angle = 0
    xa0, ya0, xa1, ya1 = bbA * 256
    xb0, yb0, xb1, yb1 = bbB * 256

    # Remove 1 pixel wide wall again to get a line
    if xa1 - xa0 == 1:
        xa1 = xa0
    if ya1 - ya0 == 1:
        ya1 = ya0
    if xb1 - xb0 == 1:
        xb1 = xb0
    if yb1 - yb0 == 1:
        yb1 = yb0
    
    dxa = xa1 - xa0
    dya = ya1 - ya0
    dxb = xb1 - xb0
    dyb = yb1 - yb0

    angleA = np.degrees(np.arctan2(dya, dxa))
    angleB = np.degrees(np.arctan2(dyb, dxb))
    angle = angleA - angleB

    if angle < -179:
        angle = angle + 360
    if angle > 180:
        angle = angle - 360

    return angle

def find_approximate_centroid(room_idx, house_edges, house_edge_adjacencies):
    # room_idx = data[house_nr][0].index(room_type)
    room_edge_ids = [id for id, edge in enumerate(house_edge_adjacencies) if room_idx in edge]
    room_edges = np.array(house_edges)[room_edge_ids]
    # Weight each edge by it's length
    weights = np.linalg.norm(room_edges[:,[2,3]] - room_edges[:,[0,1]], axis=1)**1.5
    # Uncomment below to remove weights
    # weights = np.ones(len(room_edges))
    # print(f"weights:\n {weights}")
    # print(f"roomedges:\n {room_edges}")
    x = np.concatenate([room_edges[:,0].reshape(-1,1), room_edges[:,2].reshape(-1,1)], axis=1)
    x_avg = np.mean(x, axis=1).reshape(-1,1)
    y = np.concatenate([room_edges[:,1].reshape(-1,1), room_edges[:,3].reshape(-1,1)], axis=1)
    y_avg = np.mean(y, axis=1).reshape(-1,1)
    room_edge_midpoints = np.concatenate((x_avg, y_avg), axis=1)
    # print(f"room_edge_midpoints:\n {room_edge_midpoints}")
    room_x, room_y = np.average(room_edge_midpoints, axis = 0, weights=weights)
    return room_x, room_y

house_nr = 0

def swap(vars):
    vars = vars.copy()
    vars[:,[0,1]] = vars[:,[2,3]]
    return vars

new_data = []

abc = 0 # used for restricting amount of homes processed

for home in data[1:]:

    # # Retrieve house-specific data
    # room_types = np.array(home[0])
    # rooms = np.array(home[1])
    # edges = np.array(home[2])[:,0:4]
    # edge_adjacencies = home[3]
    # doors = np.array(home[4])

    # # Plotting
    # fig, ax = plt.subplots()
    # ax.set_title("House with walls, Red ones have doors")
    # for num, edge in enumerate(edges):
    #     x = np.array([edge[0], edge[2]])
    #     x_avg = np.mean(x)
    #     y = np.array([edge[1], edge[3]])
    #     y_avg = np.mean(y)
    #     if num in doors:
    #         ax.plot(x,y, "r")
    #         plt.scatter(x_avg, y_avg, c="#FF0000")
    #     else:
    #         ax.plot(x,y, "b")
    #         plt.scatter(x_avg, y_avg, c="#0000FF")
    # for room_idx, room_type in enumerate(room_types):
    #     center_x, center_y = find_approximate_centroid(room_idx, edges, edge_adjacencies)
    #     plt.text(center_x+4, center_y-3, ROOM_CLASS[room_types[room_idx]])
    #     plt.scatter(center_x, center_y, c="#000000")
    # ax.set_aspect('equal')
    # for bb in rooms:
    #     x0, y0, x1, y1 = bb
    #     height = y1-y0
    #     width = x1-x0
    #     ax.add_patch(plt.Rectangle((x0,y0), width, height), color="red")
    

    # Creating the node list (nodes) for later use and the list of one-hot encoded node vectors (nds), (Nx(11+2))
    rooms = home[0]

    nds = one_hot_embedding(rooms)
    nds = torch.FloatTensor(nds)
    nds = torch.cat((nds, torch.FloatTensor([[-1,-1] for i in range(len(rooms))])), 1) # Add features [-1,-1] to room nodes

    # Creating the bounding boxes for the rooms (Nx4)
    bbs = np.array([bb.tolist() for bb in home[1]]) / 256 # Values between 0 and 1


    # Find Exterior Walls and add to nodes (with features length and door) and bounding boxes list
    exterior_walls = get_exterior_walls(home[3], np.array(home[2])[:,:4], home[4]) # gives edge bb np.array(home[2])[i] on place 1 (out of [0,1,2])
    ex_wall_nodes = []
    ex_wall_bbs = []
    for nod in exterior_walls:
        ex_wall_nodes.append([nod[0] + len(nds), edge_length(nod[1]), nod[2]])
        ex_wall_bbs.append(give_edge_width_1(nod[1]))
    ex_wall_nodes = np.array(ex_wall_nodes)
    ex_wall_bbs = np.array(ex_wall_bbs) / 256 # Bounding boxes of Exterior Walls values scaled between 0 and 1

    # '''
    # Plot EW bboxes
    # '''
    # for ew_bb in ex_wall_bbs:
    #     x0, y0, x1, y1 = ew_bb
    #     height = y1-y0
    #     width = x1-x0
    #     ax.add_patch(plt.Rectangle((x0,y0), width, height))
    # plt.show()  

    ew_nds = one_hot_embedding([0 for i in exterior_walls]) # Create nodes of type 0 for Exterior Walls
    ew_nds = torch.FloatTensor(ew_nds)
    ew_nds = torch.cat((ew_nds, torch.FloatTensor(ex_wall_nodes[:,1:])), 1) # Add EW node features to list

    nds = torch.cat((nds, ew_nds), 0) # Add room nodes and Exterior Wall nodes to the same nds node list
    

    bbs = np.concatenate((bbs, ex_wall_bbs), 0) # Add bounding boxes of EW to bounding boxes list


    # Creating the edges (Ex3), [src_node_id, +1/-1, dest_node_id]
    triples = []
    for k in range(len(nds)):     # From each node
        for l in range(len(nds)): # To each node
            if k != l:
                if k > len(rooms)-1 and l > len(rooms)-1: # If both k and l are EW
                    if is_adjacent(bbs[k], bbs[l]):
                        triples.append([k,1,l])
                    else:
                        triples.append([k,-1,l])

                if k > len(rooms)-1 and l <= len(rooms)-1: # if k is EW and l is room
                    if room_edge_adjacent(l, k, home[3], bbs):
                        triples.append([k,1,l])
                    else:
                        triples.append([k,-1,l])

                if k <= len(rooms)-1 and l > len(rooms)-1: # if k is room and l is EW
                    if room_edge_adjacent(k, l, home[3], bbs):
                        triples.append([k,1,l])
                    else:
                        triples.append([k,-1,l])

                if k <= len(rooms)-1 and l <= len(rooms)-1: # if both k and l are room
                    if rooms_adjacent(k, l, home[3]):
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
    For each edge [from, connected, to], if from or to is a room: edge type is RA (1) with door feature (0/1) and relative direction (E/NE/N/NW/W/SW/S/SE/Undefined)=(0/1/2/3/4/5/6/7/8)
                                    if from and to are Exterior Wall: edge type is CE (0) with angle feature in degrees and 0.
    '''
    i=0
    for edge in eds:
        if edge[0] not in range(len(rooms)) and edge[2] not in range(len(rooms)):  # Connection between two EW
            edges_f[i][0] = 0 # Edge type is CE
            edges_f[i][1] = find_angle_EW(bbs[edge[0]], bbs[edge[2]])

        else: # Connection is RA
            edges_f[i][0] = 1
            edges_f[i][2] = relative_direction(bbs[edge[0]], bbs[edge[2]]) # Find relative direction of node 1 to node 2 ################################################ Might need to me 8 for non adjacency of nodes

            if edge[0] in range(len(rooms)) and  edge[2] in range(len(rooms)): # If both nodes are rooms
                if edge[1] == 1:
                    if edge_has_door(edge[0], edge[2], home[3], home[4]): # If there is a door connecting the rooms
                        edges_f[i][1] = 1   
        i = i + 1  

    edges_f = np.array(edges_f)



    # # This converts bounding boxes into masks that fit the needed image size
    # im_size = 64
    # rooms_mks = np.zeros((len(nds), im_size, im_size)) # Dim = (# nodes, image size, image size)
    # for k, (rm, bb) in enumerate(zip(nodes, bbs)):
    #   if rm > 0:
    #     x0, y0, x1, y1 = im_size*bb
    #     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    #     rooms_mks[k, x0:x1+1, y0:y1+1] = 1.0
    # rooms_mks = torch.FloatTensor(rooms_mks)
    
    
    print('home = ')
    print(home)
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
    if abc == 1: # amount of rooms we want in the list
        break



print(new_data)

# Finally, save the list:

# np.save(new_data_path, new_data)
# #######################################
# Gives an error because the lists nds, bbs, eds and edges_f are not the same shape. Can not be put into an np.array. Perhaps try in CSV format?...
# #######################################