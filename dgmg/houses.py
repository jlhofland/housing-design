import os
import pickle
import random
import dgl
import time
import torch

import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset



def check_house(g):

    # Assert that each exterior wall is connected to at least one other room besides its single outgoing connecting wall edge
    print("\nHouse complete, checking")
    for src in range(g.num_nodes("exterior_wall")):
        total_out_degrees = 0
        for cet in g.canonical_etypes:
            if cet[0] != "exterior_wall":
                continue
            if cet[1] == "corner_edge" and cet[2] == "exterior_wall":
                if g.out_degrees(u=src, etype=cet) != 1: return False
                total_out_degrees +=1
                continue
            if cet[1] == "corner_edge" and cet[2] != "exterior_wall":
                if g.out_degrees(u=src, etype=cet) != 0: return False
                continue
            out_degrees = g.out_degrees(u=src, etype=cet)
            if out_degrees > 0:
                total_out_degrees += out_degrees
                print(f"Src ID: {src}, Out degree: {out_degrees}, ET: {cet}")
        if not total_out_degrees > 2: 
            print("One or more Exterior Walls do not connect to another Exterior Wall and minimum one other room")
            return False
        
    # Assert that each room is connected to at least two other rooms / walls (how could it be any other way)
    room_types_sans_EW = g.ntypes
    room_types_sans_EW.remove("exterior_wall")
    for room_type in room_types_sans_EW:
        for src in range(g.num_nodes(room_type)):
            total_out_degrees = 0
            for cet in g.canonical_etypes:
                if cet[1] != "corner_edge":
                    continue
                out_degrees = g.out_degrees(u=src, etype=cet)
                if out_degrees > 0:
                    total_out_degrees += out_degrees
                    print(f"Src ID: {src}, Out degree: {out_degrees}, ET: {cet}")
            if not total_out_degrees > 2: 
                print("One or more Rooms do not connect to minimum 2 other rooms")
                return False
    
    return True

def generate_home_dataset(g, num_homes):
    import random
    import torch
    import time
        
    node_types = ["exterior_wall", "living_room", "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room", "stop"]
    homes = []
    max_num_nodes = 10

    path_p = "houses_dataset.p"
    path_txt = "houses_dataset.txt"

    if os.path.exists(path_p):
        os.remove(path_p)
    if os.path.exists(path_txt):
        os.remove(path_txt)

    
    while len(homes)<num_homes:
        # Available node types
        node_counts = {"exterior_wall":0, "living_room":0, "kitchen":0, "bedroom":0, "bathroom":0, "missing":0, "closet":0, "balcony":0, "corridor":0, "dining_room":0, "laundry_room":0}
        total_num_nodes = 0

        for ntype in g.ntypes:
            node_counts[ntype] = g.num_nodes(ntype)
            total_num_nodes += g.num_nodes(ntype)
        print("####################################")
        print(f"House #: {len(homes)}")
        print(f"Initial num nodes: {total_num_nodes}")

        # Initialize the decision list
        decisions = []

        # Step 1: Add the initial node or terminate the generation process (i=0)
        initial_node_type = random.choices(node_types[1:-1], weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], k=1)[0]  # Choose a random initial node type
        decisions.append((0, initial_node_type, -1))
        stop = bool(initial_node_type == "stop")
        if not stop:
            node_counts[initial_node_type] += 1
            total_num_nodes += 1
            print(f"Added initial node of type {initial_node_type}, ID# {node_counts[initial_node_type]}. Num_nodes: {total_num_nodes}")

        else:
            print("Stopped at initial node")

        src_node_type = initial_node_type
        src_node_id = node_counts[initial_node_type]-1
        # Continue generating the house graph
        while not stop:
            # Step 2: Decide whether to add an edge or terminate (i=1)
            if decisions[-1][0] == 0:
                add_edge = 0
            else:
                add_edge = random.choice([0, 1])#, weights=[0.3, 0.7], k=1)[0]  # 0 for adding an edge, 1 for termination
            # ADDING EDGES
            decisions.append((1, add_edge, -1))
            
            if add_edge == 0:
                # Step 3: Add an edge to a destination node (i=2)
                non_zero_dests = [key for key in node_counts if node_counts[key]>0]
                destination_node_type = random.choice(non_zero_dests)  # Choose a random destination node type
                destination_node_id = random.randint(0, node_counts[destination_node_type] - 1)  # Choose a random destination node id 
                while (src_node_type, src_node_id) == (destination_node_type, destination_node_id):
                    destination_node_type = random.choice(non_zero_dests)  # Choose a random destination node type
                    destination_node_id = random.randint(0, node_counts[destination_node_type] - 1)  # Choose a random destination node id 
                edge_feature_vector = torch.tensor([[random.randint(0, 2), random.randint(0, 8)]])
                # ADDING DESTINATIONS
                decisions.append((2, (destination_node_type, destination_node_id), edge_feature_vector))
                print("Added edge")
            else:
                if total_num_nodes < max_num_nodes:
                    new_node_type = random.choice(node_types[1:])#, weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5], k=1)[0]  # Choose a random initial node type
                    stop = bool(new_node_type == "stop")
                    if not stop:
                        # ADDING NODES
                        decisions.append((0, new_node_type, -1))
                        node_counts[new_node_type] += 1
                        total_num_nodes += 1
                        src_node_type = new_node_type
                        src_node_id = node_counts[new_node_type]-1
                        print(f"Added new node of type {new_node_type}, ID# {node_counts[new_node_type]}. Num_nodes: {total_num_nodes}")
                    else:
                        decisions.append((0, "stop", -1))
                        print("Stopped at later node 1")
                else:
                    stop = True
                    decisions.append((0, "stop", -1))
                    print("Stopped at later node 2")
        homes.append(decisions)
    with open(path_p, "wb+") as f:
        pickle.dump(homes, f)
    # # Uncomment to also create a text version for inspection
    # with open(path_txt, "w") as f:
    #     for home in homes:
    #         f.write("###################\nNew home\n")
    #         for step in home:
    #             f.write(str(step)+"\n")

    # End the decision list

class HouseDataset(Dataset):
    def __init__(self, fname):
        super(HouseDataset, self).__init__()

        with open(fname, "rb") as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_single(self, batch):
        assert len(batch) == 1, "Currently we do not support batched training"
        return batch[0]

    def collate_batch(self, batch):
        return batch

class HouseModelEvaluation(object):

    # Generates new graphs, makes simple graph-validity checks, keeps track of these metrics, and plots groups of four graphs

    def __init__(self, v_min, v_max, dir):
        super(HouseModelEvaluation, self).__init__()

        self.v_min = v_min
        self.v_max = v_max

        self.dir = dir

    def assign_node_labels_and_colors(self, g):
        # color_dict_new = {"exterior_wall", "living_room", "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room"}
        # color_dict_old = {0:'black', 1:'blue', 2:'yellow', 3:'green', 4:'teal'}
        color_dict = {
            "exterior_wall": "lightblue",
            "living_room": "red",
            "kitchen": "orange",
            "bedroom": "purple",
            "bathroom": "pink",
            "missing": "gray",
            "closet": "brown",
            "balcony": "lime",
            "corridor": "cyan",
            "dining_room": "gold",
            "laundry_room": "magenta",
        }
        colors = []
        labels = {}

        # Get node-type order
        node_type_order = g.ntypes

        # Create node-type subgraph
        g_homo = dgl.to_homogeneous(g)
        # print(g_homo.ndata)

        for idx, node in enumerate(g_homo.ndata[dgl.NTYPE]):
            labels[idx] = node_type_order[node] + "_" + str(int(g_homo.ndata[dgl.NID][idx]))
            colors.append(color_dict[node_type_order[node]])
        
        return labels, colors
            

    def rollout_and_examine(self, model, num_samples):
        assert not model.training, "You need to call model.eval()."

        num_total_size = 0
        num_valid_size = 0
        num_house = 0
        num_valid = 0
        plot_times = 0
        # adj_lists_to_plot = []
        graphs_to_plot = []

        options = {
            'node_size': 300,
            'width': 1,
            'with_labels': True,
            'font_size': 12,
            'font_color': 'r',
        }

        for i in range(num_samples):
            sampled_graph = model()
            if isinstance(sampled_graph, list):
                # When the model is a batched implementation, a list of
                # DGLGraph objects is returned. Note that with model(),
                # we generate a single graph as with the non-batched
                # implementation. We actually support batched generation
                # during the inference so feel free to modify the code.
                sampled_graph = sampled_graph[0]

            graphs_to_plot.append(sampled_graph)

            graph_size = sampled_graph.num_nodes()
            valid_size = self.v_min <= graph_size <= self.v_max
            house = check_house(sampled_graph)

            num_total_size += graph_size

            if valid_size:
                num_valid_size += 1

            if house:
                num_house += 1
            else:
                print("House failed.")

            if valid_size and house:
                num_valid += 1

            if len(graphs_to_plot) >= 1:
                plot_times += 1
                fig, ax = plt.subplots(1, 1, figsize=(15,7))
                g = graphs_to_plot[0]
                labels, colors = self.assign_node_labels_and_colors(g)
                G = dgl.to_networkx(dgl.to_homogeneous(g))
                nx.draw(G, ax=ax, node_color=colors, labels=labels, **options)
                plt.savefig(self.dir + "/samples/{:d}".format(plot_times))
                plt.close()

                graphs_to_plot = []


        self.num_samples_examined = num_samples
        self.average_size = num_total_size / num_samples
        self.valid_size_ratio = num_valid_size / num_samples
        self.house_ratio = num_house / num_samples
        self.valid_ratio = num_valid / num_samples

    def write_summary(self):
        def _format_value(v):
            if isinstance(v, float):
                return "{:.4f}".format(v)
            elif isinstance(v, int):
                return "{:d}".format(v)
            else:
                return "{}".format(v)

        statistics = {
            "num_samples": self.num_samples_examined,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "average_size": self.average_size,
            "valid_size_ratio": self.valid_size_ratio,
            "house_ratio": self.house_ratio,
            "valid_ratio": self.valid_ratio,
        }

        model_eval_path = os.path.join(self.dir, "model_eval.txt")

        print("\nModel evaluation summary:")
        with open(model_eval_path, "w") as f:
            for key, value in statistics.items():
                msg = "{}\t{}\n".format(key, _format_value(value))
                f.write(msg)
                print(msg)

        print("\nSaved model evaluation statistics to {}".format(model_eval_path))

class HousePrinting(object):

    # Prints data during training

    def __init__(self, num_epochs, num_batches):
        super(HousePrinting, self).__init__()

        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_count = 0

    def update(self, epoch, metrics):
        self.batch_count = (self.batch_count) % self.num_batches + 1

        msg = "epoch {:d}/{:d}, batch {:d}/{:d}".format(
            epoch, self.num_epochs, self.batch_count, self.num_batches
        )
        for key, value in metrics.items():
            msg += ", {}: {:4f}".format(key, value)
        print(msg)
