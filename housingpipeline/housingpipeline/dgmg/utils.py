import datetime
import os
import random
import dgl
import numpy as np
import math
from pprint import pprint
import json

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init

########################################################################################################################
#                                                    configuration                                                     #
########################################################################################################################


def mkdir_p(path):
    import errno

    try:
        os.makedirs(path)
        print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Directory {} already exists.".format(path))
        else:
            raise


def date_filename(base_dir="./"):
    dt = datetime.datetime.now()
    return os.path.join(
        base_dir,
        "{}_{:02d}-{:02d}-{:02d}".format(
            dt.date(), dt.hour, dt.minute, dt.second
        ),
    )


def setup_log_dir(opts):
    log_dir = "{}".format(date_filename(opts["log_dir"]))
    mkdir_p(log_dir)
    return log_dir


def save_arg_dict(opts, filename="settings.txt"):
    def _format_value(v):
        if isinstance(v, float):
            return "{:.4f}".format(v)
        elif isinstance(v, int):
            return "{:d}".format(v)
        else:
            return "{}".format(v)

    save_path = os.path.join(opts["log_dir"], filename)
    with open(save_path, "w") as f:
        for key, value in opts.items():
            f.write("{}\t{}\n".format(key, _format_value(value)))
    print("Saved settings to {}".format(save_path))


def setup(args):
    opts = args.__dict__.copy()

    cudnn.benchmark = False
    cudnn.deterministic = True

    # Seed
    if opts["seed"] is None:
        opts["seed"] = random.randint(1, 10000)
    random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    # Dataset
    from configure import dataset_based_configure

    opts = dataset_based_configure(opts)

    assert (
        opts["path_to_dataset"] is not None
    ), "Expect path to dataset to be set."
    if not os.path.exists(opts["path_to_dataset"]):
        if opts["dataset"] == "cycles":
            raise ValueError("Cycles dataset no longer supported")
        elif opts["dataset"] == "houses":
            print("WE GOT HOUSES GIRL")
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

    # Optimization
    if opts["clip_grad"]:
        assert (
            opts["clip_grad"] is not None
        ), "Expect the gradient norm constraint to be set."

    # Log
    print("Prepare logging directory...")
    log_dir = setup_log_dir(opts)
    opts["log_dir"] = log_dir
    mkdir_p(log_dir + "/samples")

    plt.switch_backend("Agg")

    save_arg_dict(opts)
    pprint(opts)

    return opts


########################################################################################################################
#                                                         model                                                        #
########################################################################################################################

def parse_input_json(file_path):
    import json
    if type(file_path) is tuple:
        file_path = file_path[0]
    # Read in input data from JSONC file, "./input.jsonc"
    with open(file_path, 'r') as openfile:
        # Reading from json file into python dict
        layout = json.load(openfile)

    # Manually parse numerical/boolean data into tensor
    room_number_data = torch.zeros(6, dtype=torch.float32)
    room_number_data[0] = layout["number_of_living_rooms"]
    room_number_data[1] = int(layout["living_rooms_plus?"])
    room_number_data[2] = layout["number_of_bedrooms"]
    room_number_data[3] = int(layout["bedrooms_plus?"])
    room_number_data[4] = layout["number_of_bathrooms"]
    room_number_data[5] = int(layout["bathrooms_plus?"])

    # Parse walls / connections into tensors
    exterior_walls_sequence = torch.tensor(layout["exterior_walls"], dtype=torch.float32)
    connections_corners = torch.LongTensor(layout["connections_corners"])[:, 0:4]
        # Add reverse corner edges
    num_corners = connections_corners.shape[0]
    connections_corners = connections_corners.repeat(2,1)
    connections_corners[num_corners:, [1,3]] = connections_corners[num_corners:, [3,1]]
    connections_rooms = torch.LongTensor(layout["connections_rooms"])
        # Adding bi-directional corner edges (one for CW and one for CCW edges)
    corner_type_edge_features_cw = torch.tensor(layout["connections_corners"], dtype=torch.float32)[:, 4:].reshape(-1,2)
    corner_type_edge_features_ccw = corner_type_edge_features_cw.clone()
    corner_type_edge_features_ccw[:,0] = -corner_type_edge_features_ccw[:,0]
    corner_type_edge_features = torch.cat([corner_type_edge_features_cw, corner_type_edge_features_ccw], dim=0)
    corner_type_edge_features[:,1]  = torch.zeros((corner_type_edge_features.size()[0]))

    return room_number_data, exterior_walls_sequence, connections_corners, connections_rooms, corner_type_edge_features


def weights_init(m):
    """
    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def dgmg_message_weight_init(m):
    """
    This is similar as the function above where we initialize linear layers from a normal distribution with std
    1./10 as suggested by the author. This should only be used for the message passing functions, i.e. fe's in the
    paper.
    """

    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1.0 / 10)
            init.normal_(m.bias.data, std=1.0 / 10)
        else:
            raise ValueError("Expected the input to be of type nn.Linear!")

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(_weight_init)
    else:
        m.apply(_weight_init)


class Printer(object):
    def __init__(self, num_epochs, dataset_size, batch_size, writer=None):
        """Wrapper to track the learning progress.

        Parameters
        ----------
        num_epochs : int
            Number of epochs for training
        dataset_size : int
        batch_size : int
        writer : None or SummaryWriter
            If not None, tensorboard will be used to visualize learning curves.
        """
        super(Printer, self).__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_batches = math.ceil(dataset_size / batch_size)
        self.count = 0
        self.batch_count = 0
        self.writer = writer
        self._reset()

    def _reset(self):
        """Reset when an epoch is completed."""
        self.batch_loss = 0
        self.batch_prob = 0

    def _get_current_batch(self):
        """Get current batch index."""
        remainer = self.batch_count % self.num_batches
        if (remainer == 0):
            return self.num_batches
        else:
            return remainer

    def update(self, epoch, loss, prob):
        """Update learning progress.

        Parameters
        ----------
        epoch : int
        loss : float
        prob : float
        """
        self.count += 1
        self.batch_loss += loss
        self.batch_prob += prob

        if self.count % self.batch_size == 0:
            self.batch_count += 1
            if self.writer is not None:
                self.writer.add_scalar('train_log_prob', self.batch_loss, self.batch_count)
                self.writer.add_scalar('train_prob', self.batch_prob, self.batch_count)

            print('epoch {:d}/{:d}, batch {:d}/{:d}, averaged_loss {:.4f}, averaged_prob {:.4f}'.format(
                epoch, self.num_epochs, self._get_current_batch(),
                self.num_batches, self.batch_loss, self.batch_prob))
            self._reset()

def tensor_to_one_hot(tensor, num_classes):
    """
    Convert a torch tensor containing a pair of integers into one-hot encoded format.

    Args:
    - tensor (torch.Tensor): A torch tensor containing a pair of integers.
    - num_classes (int): The total number of classes or categories.

    Returns:
    - torch.Tensor: The one-hot encoded tensor.
    """

    # Initialize a zero-filled one-hot tensor
    one_hot = torch.zeros(tensor.shape[1], num_classes)

    # Set the corresponding indices to 1
    for id in range(tensor.shape[1]):
        one_hot[id, tensor[0][id]] = 1

    # Expand dims to keep indexing the same as for the original tensor
    return one_hot[None,:]


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
    return y[labels.to(dtype=torch.long)]


def get_bbs_from_user_input(user_input_path, nodes):
    
    def give_edge_width_1(edge):
        ''' Gives the bounding box of an edge width 1 '''
        x0, y0, x1, y1 = edge
        if x0 == x1:
            x1 = x1 + 1
        if y0 == y1:
            y1 = y1 + 1
        return [x0, y0, x1, y1]

    with open(user_input_path, "r") as file:
        data : dict = json.load(file)

    bbs = np.zeros((nodes.shape[0], 4))

    # Find Exterior Walls and add to nodes (with features length and door) and bounding boxes list
    exterior_walls = data["exterior_walls"]
    ex_wall_bbs = []
    for wall in exterior_walls:
        ex_wall_bbs.append(wall[1:-1]) # Edges have width 0 now still
    ex_wall_bbs = np.array(ex_wall_bbs) # Bounding boxes of Exterior Walls values scaled between 0 and 1

    # hope the order is right (TODO)
    ew_ids = [i for i, nd in enumerate(nodes) if nd[0]==1]

    for i, id in enumerate(ew_ids):
        bbs[id] = ex_wall_bbs[i]

    # Give edges width 1 (1/256 actually) here for each item in bbs list (leaves bounding boxes with width != 0 alone)
    for i, bb in enumerate(bbs):
        if np.all(bb==0):
            continue
        bbs[i] = give_edge_width_1(bbs[i])
    
    bbs = bbs / 256.0

    return bbs

def convert_bbs_to_masks(bbs):
    im_size = 64
    rooms_mks = np.full((bbs.shape[0], im_size, im_size), fill_value=-1)
    for k, bb in enumerate(bbs):
        if np.all(bb==0):
            continue
        x0, y0, x1, y1 = im_size * bb
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        rooms_mks[k, x0 : x1 + 1, y0 : y1 + 1] = 1.0

    rooms_mks = torch.FloatTensor(rooms_mks)
    return rooms_mks


def dgl_to_graphlist(g, user_input_path=None):
    ''' Converts dgl graph to listgraph format, outputs [nodes, edges, edge features] '''

    nodes_dict = {"exterior_wall": 0, "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
    connection_dict = {"corner_edge": 0, "room_adjacency_edge": 1}

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

    if user_input_path is not None:
        bbs = get_bbs_from_user_input(user_input_path=user_input_path, nodes=nds)

        masks = convert_bbs_to_masks(bbs)

        return [masks, nds, torch.tensor(eds), torch.tensor(eds_f)]
    else:
        return [nds, torch.tensor(eds), torch.tensor(eds_f)]

def lists_to_tuple(input_list):
    level_data = []
    for a in input_list:
        if isinstance(a, list):
            level_data.append(lists_to_tuple(a))
        else:
            level_data.append(a)
    return tuple(level_data)

def graphlist_to_tuple(graph_list):
    if not type(graph_list) is list:
        graph_list = graph_list.tolist()
    return lists_to_tuple(graph_list)



def graph_direction_distribution(graphlist):
    ''' Calculates the distributions for the number of wind directions checked per room.
    Takes graphlist and returns list with 5 entries for the percentage of 0,1,2,3,4 directions checked. '''

    num_directions_per_room = []
    for i, room in enumerate(graphlist[0]): # Check each room
        room_direction_list = []
        num_directions = 0
        North = 0
        East = 0
        South = 0
        West = 0

        num_outgoing_edges = 0
        if room[0] == 0.: # Not EW
            for j,edge in enumerate(graphlist[1]): # Check each edge
                if edge[1] == 1: # If edge exists
                    if edge[0] == i: # If edge is from given room
                        room_direction_list.append(int(graphlist[2][j][2].item()))
                        num_outgoing_edges += 1
            
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
            num_directions_per_room.append(num_directions)

    room_distribution = []
    for i in range(5):
        room_distribution.append(num_directions_per_room.count(i) / (len(num_directions_per_room)) + 1e-8)

    return room_distribution

def room_without_doors(graphlist):
    ''' Input graphlist, returns True/False whether there is a room with no doors '''

    room_without_door_counter = 0
    for i, room in enumerate(graphlist[0]): # Check each room
        room_has_door = False
        if room[0] == 0.:
            for j, edg in enumerate(graphlist[1]):
                if edg[0] == i: # If edge is from room
                    if graphlist[2][j][1] == 0: # If edge has door
                        room_has_door = True
                    if graphlist[0][edg[2]][0] == 1.: # To node is exterior wall
                        if graphlist[0][edg[2]][12] == 1:
                            room_has_door = True
            if not room_has_door:
                print(f"{room} has no door")
                room_without_door_counter += 1

    return room_without_door_counter > 0