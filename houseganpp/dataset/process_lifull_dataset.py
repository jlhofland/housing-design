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
      "nds": all graph nodes in an Nx11 list, with each node represented as a one-hot encoded vector with 11 classes. see one_hot_embedding below.
      "bbs": all graph node bounding boxes in an Nx4 list, including exterior wall nodes (need to expand the EW edges into single pixel-wide bbs).
      "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
      "nds_f": all graph node features in an Nx2 list. 
      "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]

      Note that N == number of graph nodes, and E == number of graph edges

[ [ [nds],[bbs],[eds],[nds_f],[eds_f] ],
 [ [],[],... ],
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



new_data = []
for home in data:
    '''
      TODO Exterior walls have to be added (with width 1 and node features (door or not)) (not sure if this will come from Alex's program)
      TODO Node features (which features are needed again? I thought (room_type, ???) )
      TODO Edge features (these will come from Alex's program)
      TODO Edges between exterior walls have to be added, I think they will follow from the algorithm used now, only the edge type will have to be changed.
    '''
    # Creating the node list (nodes) for later use and the list of one-hot encoded node vectors (nds), (Nx11)
    nodes = home[0]
    nds = one_hot_embedding(nodes)[:,1:]
    nds = torch.FloatTensor(nds)

    # Creating the bounding boxes (Nx4)
    bbs = np.array([bb.tolist() for bb in home[1]]) / 256 # Values between 0 and 1
    
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

    # Creating the node features list (Nx2)
    for n in range(len(nodes)):
        nds_f = [home[0][n],0] # Add node features (room type and .....?????)

    # Creating the edge features list (Ex3), [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]





    # # This converts bounding boxes into masks that fit the needed image size
    # im_size = 16
    # rooms_mks = np.zeros((len(nodes), im_size, im_size)) # Dim = (# nodes, image size, image size)
    # for k, (rm, bb) in enumerate(zip(nodes, bbs)):
    #   if rm > 0:
    #     x0, y0, x1, y1 = im_size*bb
    #     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    #     rooms_mks[k, x0:x1+1, y0:y1+1] = 1.0
    # rooms_mks = torch.FloatTensor(rooms_mks)
    
    
    

    print('home = ')
    print(home)
    print('nodes = ')
    print(nodes)
    print('bounding boxes = ')
    print(bbs)
    #print('room masks = ')
    #print(rooms_mks)
    print('triples = ')
    print(triples)
    print('edges = ')
    print(eds)
    print('nodes features = ')
    print(nds_f)

    break




# Finally, save the list:
# np.save(new_data_path, new_data)

