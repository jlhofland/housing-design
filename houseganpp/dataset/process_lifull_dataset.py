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
      "bbs": all graph node bounding boxes in an Nx4 list, including exterior wall nodes (need to expand the EW edges into single pixel-wide bbs).
      "nds": all graph nodes (types/features) in an Nx13 list, with each node represented as a one-hot encoded vector with 11 classes concatenated with two feature elements. see one_hot_embedding below.
      "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
      "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]

      Note that N == number of graph nodes, and E == number of graph edges
      Note that eval_data and test_data from from housegan_valid_data.npy, split in half between them
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

def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b

    h1, h2 = x1 - x0, x3 - x2
    w1, w2 = y1 - y0, y3 - y2

    xc1, xc2 = (x0 + x1) / 2.0, (x2 + x3) / 2.0
    yc1, yc2 = (y0 + y1) / 2.0, (y2 + y3) / 2.0

    delta_x = np.abs(xc2 - xc1) - (h1 + h2) / 2.0
    delta_y = np.abs(yc2 - yc1) - (w1 + w2) / 2.0

    delta = max(delta_x, delta_y)

    return delta < threshold

def build_graph(self, bbs, types):
		# create edges -- make order
		triples = []
		nodes = types
		bbs = np.array(bbs)
        
		# encode connections
		for k in range(len(nodes)):
			for l in range(len(nodes)):
				if True:#l > k:
					nd0, bb0 = nodes[k], bbs[k]
					nd1, bb1 = nodes[l], bbs[l]
					if is_adjacent(bb0, bb1):
						if 'train' in self.split:
							triples.append([k, 1, l])
						else:
							triples.append([k, 1, l])
					else:
						if 'train' in self.split:
							triples.append([k, -1, l])
						else:
							triples.append([k, -1, l])

		# convert to array
		nodes = np.array(nodes)
		triples = np.array(triples)
		bbs = np.array(bbs)
		return bbs, nodes, triples




original_data_path = '../datasets/housegan_clean_data.npy'
new_data_path = '../datasets/HHGPP_train_data.npy'

data = np.load(original_data_path, allow_pickle=True)

new_data = []
for home in data:
    '''
    do all the processing
    '''

# Finally, save the list:
np.save(new_data_path, new_data)

