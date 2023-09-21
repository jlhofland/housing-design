import datetime
import os
import random
import dgl
import numpy as np
from pprint import pprint

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
            from cycles import generate_dataset

            generate_dataset(
                opts["min_size"],
                opts["max_size"],
                opts["ds_size"],
                opts["path_to_dataset"],
            )
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
    # Read in input data from JSONC file, "./input.jsonc"
    with open(file_path, 'r') as openfile:
        # Reading from json file into python dict
        layout = json.load(openfile)

    # Manually parse numerical/boolean data into tensor
    room_number_data = torch.zeros(6)
    room_number_data[0] = layout["number_of_living_rooms"]
    room_number_data[1] = int(layout["living_rooms_plus?"])
    room_number_data[2] = layout["number_of_bedrooms"]
    room_number_data[3] = int(layout["bedrooms_plus?"])
    room_number_data[4] = layout["number_of_bathrooms"]
    room_number_data[5] = int(layout["bathrooms_plus?"])

    # Parse walls / connections into tensors
    exterior_walls_sequence = torch.tensor(layout["exterior_walls"], dtype=torch.float32)
    connections_corners = torch.LongTensor(layout["connections_corners"])[:, 0:4]
    connections_rooms = torch.LongTensor(layout["connections_rooms"])
    corner_type_edge_features = torch.tensor(layout["connections_corners"], dtype=torch.float32)[:, 4].reshape(-1,1)

    return room_number_data, exterior_walls_sequence, connections_corners, connections_rooms, corner_type_edge_features

def define_empty_typed_graph(ntypes, etypes):

    def remove_all_edges(g):
        for etype in g.canonical_etypes:
            num_eids = g.num_edges(etype)
            eids = list(range(num_eids))
            g.remove_edges(eids=eids, etype=etype)

    def remove_all_nodes(g):
        for ntype in g.ntypes:
            num_nids = g.num_nodes(ntype)
            nids = list(range(num_nids))
            g.remove_nodes(nids=nids, ntype=ntype)

    def empty_out_graph(g):
        remove_all_edges(g)
        remove_all_nodes(g)

    graph_data = {}
    for etype in etypes:
        for src_ntype in ntypes:
            # print(f"src: {src_ntype}")
            # print(f"ntypes: {ntypes}")
            dest_ntypes = ntypes.copy()
            # dest_ntypes.remove(str(src_ntype))
            # print(f"dest_ntypes: {dest_ntypes}")
            for dest_ntype in dest_ntypes:
                if etype == "corner_edge" and (src_ntype != "exterior_wall" or dest_ntype != "exterior_wall"):
                    continue
                canonical_etype = (src_ntype, etype, dest_ntype)
                nids = (torch.tensor([0]), torch.tensor([0]))
                graph_data[canonical_etype] = nids

    g = dgl.heterograph(graph_data)
    empty_out_graph(g)
    return g

def apply_partial_graph_input_completion(file_path, room_types, edge_types):
    # Retrieve input data
    _, exterior_walls_sequence, connections_corners_sequence, connections_rooms_sequence, corner_type_edge_features = parse_input_json(file_path=file_path)

    # Extract wall features
    exterior_walls_features = []
    for wall in exterior_walls_sequence:
        wall = wall.numpy()
        wall_start = wall[0:2]
        wall_end = wall[2:4] 
        wall_length = np.linalg.norm(wall_end - wall_start)
        exterior_walls_features.append(wall_length)
    exterior_walls_features = torch.tensor(exterior_walls_features, dtype=torch.float32).reshape(-1,1)

    # Initialize empty graph with all node and edge types pre-defined
    g = define_empty_typed_graph(room_types, edge_types)

    # Uncomment to show empty graph structure
    # for c_et in g.canonical_etypes:
    #     if g.num_edges(c_et) >= 0:
    #         print(f"ET: {c_et} : {g.num_edges(c_et)}")

    for connection in connections_corners_sequence:
        etype = (room_types[connection[0].item()], 'corner_edge', room_types[connection[2].item()])
        g.add_edges(u=connection[1].item(), v=connection[3].item(), etype=etype)
    for connection in connections_rooms_sequence:
        etype = (room_types[connection[0].item()], 'room_adjacency_edge', room_types[connection[2].item()])
        g.add_edges(u=connection[1].item(), v=connection[3].item(), etype=etype)

    # Add in wall-node features
    g.nodes['exterior_wall'].data['hf'] = exterior_walls_features
    
    # Add in corner edge features
    g.edges['corner_edge'].data['e'] = corner_type_edge_features
    
    # Add in room-adjacency edge features
    # First, initialize room_adjacency edges with garbage
    for etype in g.canonical_etypes:
        if etype == ('exterior_wall', 'corner_edge', 'exterior_wall'):
            continue
        num_etype = g.num_edges(etype)
        g.edges[etype].data['e'] = torch.tensor([99,99]).repeat(num_etype,1)
    # Then, add in actual feature
    for connection in connections_rooms_sequence:    
        etype_tuple = (room_types[connection[0].item()], 'room_adjacency_edge', room_types[connection[2].item()])
        edge_id = g.edge_ids(connection[1].item(), connection[3].item(), etype=etype_tuple)
        g.edges[etype_tuple].data['e'][edge_id] = connection[4:]

    def initializer(shape, dtype, ctx, range):
        return torch.tensor([99], dtype=dtype, device=ctx).repeat(shape)

    for ntype in g.ntypes:
        g.set_n_initializer(initializer, ntype=ntype)
    for etype in g.canonical_etypes:
        g.set_e_initializer(initializer, etype=etype)
    
    # Uncomment to examine filled graph structure
    # for c_et in g.canonical_etypes:
    #     if g.num_edges(c_et) > 0:
    #         print(f"Edge numbers: {c_et} : {g.num_edges(c_et)}")
    #         print(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}")

    return g

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
