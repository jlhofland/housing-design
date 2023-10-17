"""
This file will prepare the lifull dataset (housegan_clean_data.npy or train_data.npy/valid_data.npy) for training and validation in HeteroHouseGAN++

Inputs:
  
  A .npy file (housegan_clean_data.npy) with the following information:
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
      "bbs": all graph node bounding boxes in an Nx4 list, including exterior wall nodes (EW have all been expanded 1 pixel in the positive x or y direction). All values are scaled between 0 and 1 by dividing by 256
      "nds": all graph nodes with features in an (Nx13) list. Each node is represented as a one-hot encoded vector with 11 classes, concatinated with the node features [length, door/no door], this is [-1,-1] for room nodes.
      "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
      "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]

      Note that N == number of graph nodes, and E == number of graph edges

[ [ [nds],[bbs],[eds],[eds_f] ],
  [ [],[],[],[],[] ],
 ...]
"""
import math
import pickle
import torch
import numpy as np
import random
import matplotlib.pyplot as plt


original_data_path = 'housingpipeline\housingpipeline\houseganpp\dataset\housegan_clean_data.npy'

new_data_path_train = 'housingpipeline\housingpipeline\houseganpp\dataset\HHGPP_train_data.npy'
new_data_path_eval = 'housingpipeline\housingpipeline\houseganpp\dataset\HHGPP_eval_data.npy'
new_data_path_test = 'housingpipeline\housingpipeline\houseganpp\dataset\HHGPP_test_data.npy'


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
    ''' Returns edge length '''
    x0, y0, x1, y1 = edge
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)
    # if abs(x0 - x1) <= 3:
    #     return abs(y1-y0)
    # if abs(y0 - y1) <= 3:
    #     return abs(x1-x0)
    # else:
    #     return np.sqrt((x1-x0)^2 + (y1-y0)^2) # For edges that are at a too large slope, return adge length with pythagoras

def give_edge_width_1(edge):
    ''' Gives the bounding box of an edge width 1 '''
    x0, y0, x1, y1 = edge
    if x0 == x1:
        x1 = x1 + 1/256
    if y0 == y1:
        y1 = y1 + 1/256
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
    ''' Gives relative direction from bb1 to bb2 '''
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

def find_angle_EW(EW1, EW2):
    ''' Get the angles between EW1 and EW2 from the bounding boxes '''
    
    angle = 0
    
    x0, y0, x1, y1 = EW1
    x2, y2, x3, y3 = EW2

    # If left endpoint wall 2 on right endpoint of wall 1:
    if  (x1,y1) == (x2, y2):
        angle1 = np.degrees(np.arctan2(y1-y0, x1-x0))
        angle2 = np.degrees(np.arctan2(y3-y2, x3-x2))
        angle = angle2 - angle1

    # If right endpoint wall 2 on right endpoint of wall 1:
    if  (x1,y1) == (x3, y3):
        angle1 = np.degrees(np.arctan2(y1-y0, x1-x0))
        angle2 = np.degrees(np.arctan2(y2-y3, x2-x3)) # flipped
        angle = angle2 - angle1

    # If left endpoint wall 2 on left endpoint of wall 1:
    if  (x0,y0) == (x2, y2):
        angle1 = np.degrees(np.arctan2(y0-y1, x0-x1)) # flipped
        angle2 = np.degrees(np.arctan2(y3-y2, x3-x2))
        angle = angle2 - angle1

    # If right endpoint wall 2 on left endpoint of wall 1:
    if  (x0,y0) == (x3, y3):
        angle1 = np.degrees(np.arctan2(y0-y1, x0-x1)) # flipped
        angle2 = np.degrees(np.arctan2(y2-y3, x2-x3)) # flipped
        angle = angle2 - angle1

    if angle <= -180:
        angle = angle + 360
    if angle > 180:
        angle = angle - 360
    return angle



# # Was used for plotting
# def find_approximate_centroid(room_idx, house_edges, house_edge_adjacencies):
#     # room_idx = data[house_nr][0].index(room_type)
#     room_edge_ids = [id for id, edge in enumerate(house_edge_adjacencies) if room_idx in edge]
#     room_edges = np.array(house_edges)[room_edge_ids]
#     # Weight each edge by it's length
#     weights = np.linalg.norm(room_edges[:,[2,3]] - room_edges[:,[0,1]], axis=1)**1.5
#     # Uncomment below to remove weights
#     # weights = np.ones(len(room_edges))
#     # print(f"weights:\n {weights}")
#     # print(f"roomedges:\n {room_edges}")
#     x = np.concatenate([room_edges[:,0].reshape(-1,1), room_edges[:,2].reshape(-1,1)], axis=1)
#     x_avg = np.mean(x, axis=1).reshape(-1,1)
#     y = np.concatenate([room_edges[:,1].reshape(-1,1), room_edges[:,3].reshape(-1,1)], axis=1)
#     y_avg = np.mean(y, axis=1).reshape(-1,1)
#     room_edge_midpoints = np.concatenate((x_avg, y_avg), axis=1)
#     # print(f"room_edge_midpoints:\n {room_edge_midpoints}")
#     room_x, room_y = np.average(room_edge_midpoints, axis = 0, weights=weights)
#     return room_x, room_y

new_data = []

nr_homes = 0 # used for restricting amount of homes processed
home_data_length = len(data)
for home in data:

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
        ex_wall_bbs.append(nod[1]) # Edges have width 0 now still
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

    eds = np.array(triples)
    

    # Creating the edge features list (Ex3), [edge type (CE=0 or RA=1), Feature1, Feature2]
    # For each edge [from, connected, to], if from node and/or to node is a room: edge type is RA (1) where feature1 is a door feature (no door=0/door=1) and
    # Feature2 is the relative direction from node 1 to node 2 (E/NE/N/NW/W/SW/S/SE/Undefined)=(0/1/2/3/4/5/6/7/8).
    # if from and to are Exterior Wall: edge type is CE where Feature1 is the angle from EW1 to EW2 in degrees and Feature2 is 0.

    edges_f = [[0, 0, 0] for i in range(len(eds))] # Create list for features [0, 0, 0]

    # Create edges_f here
    i=0
    for edge in eds:
        if edge[0] not in range(len(rooms)) and edge[2] not in range(len(rooms)):  # Connection between two EW
            edges_f[i][0] = 0 # Edge type is CE
            if edge[1] == 1:
                edges_f[i][1] = find_angle_EW(bbs[edge[0]], bbs[edge[2]])

        else: # Connection is RA
            edges_f[i][0] = 1
            edges_f[i][2] = relative_direction(bbs[edge[0]], bbs[edge[2]]) # Find relative direction of node 1 to node 2

            if edge[0] in range(len(rooms)) and  edge[2] in range(len(rooms)): # If both nodes are rooms
                if edge[1] == 1:
                    if edge_has_door(edge[0], edge[2], home[3], home[4]): # If there is a door connecting the rooms
                        edges_f[i][1] = 1   
        i = i + 1  

    edges_f = np.array(edges_f)


    # # Print stuff for checking
    # print(nds)
    # print(bbs*256)
    # eee = np.hstack((eds,edges_f))
    # for e in eee:
    #     if e[3] == 0 and e[1] == 1:
    #         print(e)


    # Give edges width 1 (1/256 actually) here  for each item in bbs list (leaves bounding boxes with width != 0 alone)
    for bb in range(len(bbs)):
        bbs[bb] = give_edge_width_1(bbs[bb])

    new_data.append([bbs, nds, eds, edges_f])

    nr_homes = nr_homes + 1
    print(f'{nr_homes} out of {home_data_length}')
    # if nr_homes == 10: # amount of rooms we want in the list
    #     break



# Finally, split and save the list:

# 60-20-20

train_data = new_data[ : math.floor(0.6 * len(new_data))]
eval_data = new_data[math.floor(0.6 * len(new_data)) : math.floor(0.8 * len(new_data))]
test_data = new_data[math.floor(0.8 * len(new_data)) : ]

with open(new_data_path_train, 'wb') as f:
  pickle.dump(train_data,f)
  f.close()

with open(new_data_path_eval, 'wb') as f:
  pickle.dump(eval_data,f)
  f.close()

with open(new_data_path_test, 'wb') as f:
  pickle.dump(test_data,f)
  f.close()

