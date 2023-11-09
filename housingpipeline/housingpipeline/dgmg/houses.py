import os
import pickle
import random
import dgl
import time
import torch
import json
import wandb
import numpy as np
from collections import OrderedDict
from housingpipeline.dgmg.utils import dgl_to_graphlist, graph_direction_distribution, room_without_doors, graphlist_to_tuple


import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset

lifull_data_distribution = [0, 3.3926383920000003e-04, 5.467806213999999e-03, 6.611541502e-02, 9.280775147999998e-01]

def check_house(model, quiet=False):
    g = model.g
    issues = set()
    # results: 
    # First the user input constraints are all fully met. 
    # Second, all rooms must have at least one door. 
    # Third, all exterior walls must connect to at least two other walls and one room. 
    # Fourth, room must connect to another room and a wall, at minimum. 
    # Fifth, each room must have outgoing connections that together cover all directions.
    results = np.zeros(5)
    # Assert that each exterior wall is connected to at least one other room besides its single outgoing connecting wall edge
    print("\nHouse complete, checking")
    for src in range(g.num_nodes("exterior_wall")):
        total_out_degrees = 0
        for cet in g.canonical_etypes:
            if cet[0] != "exterior_wall":
                continue
            if cet[1] == "corner_edge" and cet[2] == "exterior_wall":
                # Each "exterior_Wall" node should have exactly two edges point to two other walls.
                if g.out_degrees(u=src, etype=cet) != 2:
                    issues.add("One or more Exterior Walls do not connect to exactly two other Exterior Walls")
                    continue
                    # print("One or more Exterior Walls do not connect to exactly two other Exterior Walls")
                    # return False
                total_out_degrees += 2
                continue
            if cet[1] == "corner_edge" and cet[2] != "exterior_wall":
                # No "corner_edge" type edges should connect an "exterior_wall" node and another room type node
                if g.out_degrees(u=src, etype=cet) != 0:
                    issues.add("Corner-type edge used to illegally connect 'exterior wall' and another room-type node")
                    continue
                    # print("Corner-type edge used to illegally connect 'exterior wall' and another room-type node")
                    # return False
                continue
            out_degrees = g.out_degrees(u=src, etype=cet)
            if out_degrees > 0:
                total_out_degrees += out_degrees
                # print(f"Exterior-Wall Src ID: {src}, Out degree: {out_degrees}, ET: {cet}")
        if not total_out_degrees > 2:
            # Each "exterior_Wall" node should have connections to at least one other room-type  
            issues.add("One or more Exterior Walls do not connect to two other Exterior Walls and minimum one other room")
            results[2] = 1
            break
            # print(
            #     "One or more Exterior Walls do not connect to two other Exterior Walls and minimum one other room"
            # )
            # return False

    # Assert that each room is connected to at least two other rooms / walls (how could it be any other way)
    room_types_sans_EW = g.ntypes.copy()
    room_types_sans_EW.remove("exterior_wall")
    for room_type in room_types_sans_EW:
        for src in range(g.num_nodes(room_type)):
            total_out_degrees = 0
            for cet in g.canonical_etypes:
                if cet[0] != room_type or cet[1] == "corner_edge":
                    continue
                out_degrees = g.out_degrees(u=src, etype=cet)
                if out_degrees > 0:
                    total_out_degrees += out_degrees
                    # print(f"Room Src ID: {src}, Out degree: {out_degrees}, ET: {cet}")
            if not total_out_degrees >= 2:
                issues.add("One or more Rooms do not connect to minimum 2 other rooms")
                results[3] = 1
                break
                # print("One or more Rooms do not connect to minimum 2 other rooms")
                # return False

    # Check user-input room number constraints
    with open(model.user_input_path, "r") as file:
        ui = json.load(file)
    
    rooms = ["living_room", "bedroom", "bathroom"]
    ui_data = [
        (ui["number_of_living_rooms"],ui["living_rooms_plus?"]),
        (ui["number_of_bedrooms"],ui["bedrooms_plus?"]),
        (ui["number_of_bathrooms"],ui["bathrooms_plus?"]),
    ]
    for i, room in enumerate(rooms):
        if model.g.num_nodes(room) < ui_data[i][0]:
            issues.add("Too few " + room + "s")
            results[0] = 1
            continue
        elif model.g.num_nodes(room) > ui_data[i][0] and ui_data[i][1] == False:
            issues.add("Too many " + room + "s")
            results[0] = 1
            continue

    # Check edge features
        # Check if feature[0] is in range(3)
        # Check if feature[1] is in range(9)
    room_types_sans_EW = g.ntypes.copy()
    room_types_sans_EW.remove("exterior_wall")
    for room_type in room_types_sans_EW:
        for src in range(g.num_nodes(room_type)):
            for cet in g.canonical_etypes:
                if cet[0] != room_type or cet[1] == "corner_edge":
                    continue
                out_degrees = g.out_degrees(u=src, etype=cet)
                if out_degrees > 0:
                    total_out_degrees += out_degrees
                    # print(f"Room Src ID: {src}, Out degree: {out_degrees}, ET: {cet}")
            if not total_out_degrees >= 2:
                issues.add("One or more Rooms do not connect to minimum 2 other rooms")
                break
    
    # Per room (not exterior wall) check feature 1's for all connected edges to confirm that 
        #   together they "bound" the room (should have one each from "N", "S", "W", and "E"), 
        #   where "N" is a list that includes "NW", "N", and "NE". We have identified that this system 
        #   is not fool-proof, in that, it is theoretically possible in the training data to have to border 
        #   rooms whose centroids are so far apart that there are no room adjaceny edges in a direction (N, S, E, W).
        #   We will therefor create a list per house that shows the percentage of rooms where there are
        #   [0,1,2,3,4] of the directions connected. We will compare this value to the overall distribution
        #   of the LIFULL dataset and check whether the found distribution is close enough to the LIFULL distribution (20%).

    graphlist = dgl_to_graphlist(g)
    graph_distribution = graph_direction_distribution(graphlist)
    
    # ground-truth distribution:[0.00000000 0.000347970985 0.00549958065 0.0662011909 0.927951257]
    total_zero_and_one = 0
    for i,percentage in enumerate(graph_distribution):
        if i > 1:
            continue
        total_zero_and_one += percentage
    if total_zero_and_one > 0.3:#0.000347970985:
    # if percentage > 1.5
        issues.add("Room direction distribution is too far off the lifull distribution")
        results[4] = 1
    
    # # Checks whether every room in the house has a door (either to another room or connected to exterior wall with a door)
    if room_without_doors(graphlist):
        issues.add("One or more rooms have no doors")
        results[1] = 1


    if not issues:
        if not quiet:
            print("House is valid.")
        return True, results
    else:
        if not quiet:
            print("House failed.")
            print("Issues with home:")
            for issue in issues:
                if issue == "Room direction distribution is too far off the lifull distribution":
                    print(f"{issue}: {graph_distribution}")
                else:
                    print(issue)

        return False, results


def generate_home_dataset(g, num_homes):
    import random
    import torch
    import time

    node_types = [
        "exterior_wall",
        "living_room",
        "kitchen",
        "bedroom",
        "bathroom",
        "missing",
        "closet",
        "balcony",
        "corridor",
        "dining_room",
        "laundry_room",
        "stop",
    ]
    homes = []
    max_num_nodes = 10

    path_p = "houses_dataset.p"
    path_txt = "houses_dataset.txt"

    if os.path.exists(path_p):
        os.remove(path_p)
    if os.path.exists(path_txt):
        os.remove(path_txt)

    while len(homes) < num_homes:
        # Available node types
        node_counts = {
            "exterior_wall": 0,
            "living_room": 0,
            "kitchen": 0,
            "bedroom": 0,
            "bathroom": 0,
            "missing": 0,
            "closet": 0,
            "balcony": 0,
            "corridor": 0,
            "dining_room": 0,
            "laundry_room": 0,
        }
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
        initial_node_type = random.choices(
            node_types[1:-1],
            weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            k=1,
        )[
            0
        ]  # Choose a random initial node type
        decisions.append((0, initial_node_type, -1))
        stop = bool(initial_node_type == "stop")
        if not stop:
            node_counts[initial_node_type] += 1
            total_num_nodes += 1
            print(
                f"Added initial node of type {initial_node_type}, ID# {node_counts[initial_node_type]}. Num_nodes: {total_num_nodes}"
            )

        else:
            print("Stopped at initial node")

        src_node_type = initial_node_type
        src_node_id = node_counts[initial_node_type] - 1
        # Continue generating the house graph
        while not stop:
            # Step 2: Decide whether to add an edge or terminate (i=1)
            if decisions[-1][0] == 0:
                add_edge = 0
            else:
                add_edge = random.choice(
                    [0, 1]
                )  # , weights=[0.3, 0.7], k=1)[0]  # 0 for adding an edge, 1 for termination
            # ADDING EDGES
            decisions.append((1, add_edge, -1))

            if add_edge == 0:
                # Step 3: Add an edge to a destination node (i=2)
                non_zero_dests = [key for key in node_counts if node_counts[key] > 0]
                destination_node_type = random.choice(
                    non_zero_dests
                )  # Choose a random destination node type
                destination_node_id = random.randint(
                    0, node_counts[destination_node_type] - 1
                )  # Choose a random destination node id
                while (src_node_type, src_node_id) == (
                    destination_node_type,
                    destination_node_id,
                ):
                    destination_node_type = random.choice(
                        non_zero_dests
                    )  # Choose a random destination node type
                    destination_node_id = random.randint(
                        0, node_counts[destination_node_type] - 1
                    )  # Choose a random destination node id
                edge_feature_vector = torch.tensor(
                    [[random.randint(0, 2), random.randint(0, 8)]]
                )
                # ADDING DESTINATIONS
                decisions.append(
                    (
                        2,
                        (destination_node_type, destination_node_id),
                        edge_feature_vector,
                    )
                )
                print("Added edge")
            else:
                if total_num_nodes < max_num_nodes:
                    new_node_type = random.choice(
                        node_types[1:]
                    )  # , weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5], k=1)[0]  # Choose a random initial node type
                    stop = bool(new_node_type == "stop")
                    if not stop:
                        # ADDING NODES
                        decisions.append((0, new_node_type, -1))
                        node_counts[new_node_type] += 1
                        total_num_nodes += 1
                        src_node_type = new_node_type
                        src_node_id = node_counts[new_node_type] - 1
                        print(
                            f"Added new node of type {new_node_type}, ID# {node_counts[new_node_type]}. Num_nodes: {total_num_nodes}"
                        )
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


class CustomDataset(Dataset):
    def __init__(self, user_input_folder, partial_seq_path=None, complete_seq_path=None, eval_only=False):
        super(CustomDataset, self).__init__()

        self.eval_only = eval_only
        self.file_names = {}
        self.files = []
        with os.scandir(user_input_folder) as dir:
            for entry in dir:
                self.file_names[int(entry.name[:-5])] = entry.path

        self.files = list(OrderedDict(sorted(self.file_names.items())).values())
        
        if not self.eval_only:
            self.partial_seq = None
            self.complete_seq = None

            with open(partial_seq_path, "rb") as partial:
                self.partial_seq = pickle.load(partial)

            with open(complete_seq_path, "rb") as complete:
                self.complete_seq = pickle.load(complete)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        user_input_path = self.files[index]
        # user_input = None
        # with open(self.files[index], "rb") as input:
        #     user_input = json.load(input)
        if not self.eval_only:
            partial_seq = self.partial_seq[index]
            complete_seq = self.complete_seq[index]
            return (user_input_path, partial_seq, complete_seq)
        else:
            return user_input_path

    def collate_single(self, batch):
        assert len(batch) == 1, "Currently we do not support batched training"
        return batch[0]

    def collate_batch(self, batch):
        return batch

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


class UserInputDataset(Dataset):
    def __init__(self, fname):
        super(UserInputDataset, self).__init__()
        self.index = 0
        self.file_names = {}
        with os.scandir(fname) as dir:
            for entry in dir:
                self.file_names[int(entry.name[:-5])] = entry.path
            self.file_names = list(OrderedDict(sorted(self.file_names.items())).values())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        return self.file_names[index]
        

class HouseModelEvaluation(object):
    # Generates new graphs, makes simple graph-validity checks, keeps track of these metrics, and plots groups of four graphs

    def __init__(self, v_min, v_max, dir):
        super(HouseModelEvaluation, self).__init__()

        self.v_min = v_min
        self.v_max = v_max

        self.dir = dir



    def generate_single_valid_graph(self, model):
        assert not model.training, "You need to call model.eval()."
        found_one = False
        while not found_one:
            sampled_graph = model.forward_pipeline(user_interface=True)
            found_one, _ = check_house(model)
        return sampled_graph

    
    def rollout_and_examine(self, model, num_samples, epoch=None, eval_it=None, data_it=None, run=None, lifull_num=None):
        try:    
            assert not model.training, "You need to call model.eval()."

            num_total_size = 0
            num_valid_size = 0
            num_house = 0
            num_valid = 0
            plot_times = 0
            graphs_to_plot = []
            total_results = np.zeros((1,5))

            options = {
                "node_size": 300,
                "width": 1,
                "with_labels": True,
                "font_size": 12,
                "font_color": "r",
            }

            print(f"Evaluation saving to {self.dir}")

            hashed_graphs = []
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

                graph_list = dgl_to_graphlist(sampled_graph)
                graph_tuple = graphlist_to_tuple(graph_list)
                graph_hash = hash(graph_tuple)
                if graph_hash not in hashed_graphs:
                    hashed_graphs.append(graph_hash)

                if epoch is None:
                    dgl.save_graphs(f"./eval_graphs/{lifull_num}/dgmg_eval_{eval_it}_graph_{i}.bin", [sampled_graph])

                    # Uncomment to examine filled graph structure
                    with open(f"./eval_graphs/{lifull_num}/graph_data_eval_{eval_it}_graph_{i}.txt", "w") as file:
                        file.write(f"Graph {i}\n\n")
                        for c_et in sampled_graph.canonical_etypes:
                            if sampled_graph.num_edges(c_et) > 0:
                                file.write(f"Edge numbers: {c_et} : {sampled_graph.num_edges(c_et)}\n")
                                file.write(f"Edge features: {c_et} :\n {sampled_graph.edges[c_et].data['e']}\n")
                        for nt in sampled_graph.ntypes:
                            if sampled_graph.num_nodes(nt) > 0:
                                file.write(f"Node features: {nt} :\n {sampled_graph.nodes[nt].data}\n")


                graph_size = sampled_graph.num_nodes()
                valid_size = self.v_min <= graph_size <= self.v_max
                house, results = check_house(model)
                total_results += results

                num_total_size += graph_size

                if valid_size:
                    num_valid_size += 1

                if house:
                    num_house += 1
                    print("House passed!")
                else:
                    print("House failed.. " + '"_"')

                if valid_size and house:
                    num_valid += 1


            if len(graphs_to_plot) >= 1:
                plot_times += 1
                fig, ax = plt.subplots(1, 1, figsize=(15, 7))
                g = graphs_to_plot[0]
                labels, colors = assign_node_labels_and_colors(g)
                G = dgl.to_networkx(dgl.to_homogeneous(g.cpu()))
                nx.draw(G, ax=ax, node_color=colors, labels=labels, **options)
                os.makedirs(self.dir, exist_ok=True)
                os.makedirs(self.dir + f"/samples/epoch_{epoch}/eval_{eval_it}/", exist_ok=True)
                if epoch is not None:
                    plt.savefig(self.dir + "/samples/epoch_{:d}/eval_{:d}/data_{:d}_gen_{:d}.png".format(epoch, eval_it, data_it, plot_times))
                    plt.close()
                    run.save(self.dir + "/samples/epoch_{:d}/eval_{:d}/data_{:d}_gen_{:d}.png".format(epoch, eval_it, data_it, plot_times))
                else:
                    plt.savefig(self.dir + "/samples_{:d}.png".format(plot_times))
                    if run:
                        run.save(self.dir + "/samples_{:d}.png".format(plot_times))

                    plt.close()
                    # graphs_to_plot.append(model.g)
                # dgl.save_graphs("./example_graphs/dgmg_graph_"+str(i)+".bin", [model.g])

                if epoch is None:
                    plot_eval_graphs(f"./eval_graphs/{lifull_num}/", graphs_to_plot, eval_it)

                graphs_to_plot = []

            self.num_samples_examined = num_samples
            self.average_size = num_total_size / num_samples
            self.valid_size_ratio = num_valid_size / num_samples
            self.house_ratio = num_house / num_samples
            self.valid_ratio = num_valid / num_samples
            self.total_results = total_results / num_samples
            self.novel_ratio = len(hashed_graphs) / num_samples
        except Exception as e:
            print(f"Rollout error... {e}")

    def write_summary(self, epoch=None, eval_it=None, data_it=None, run=None, cli_only=False, lifull_num=None):
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
            "valid_ratio": self.valid_ratio,
            "validity_results": self.total_results,
            "novel_ratio": self.novel_ratio
        }

        try:
            if epoch is not None:
                model_eval_path = os.path.join(self.dir, f"model_eval_epoch_{epoch}_eval_{eval_it}_data_{data_it}.txt")
            else:
                model_eval_path = os.path.join(f"./eval_graphs/{lifull_num}/", f"model_eval.txt")


            print("\nModel evaluation summary:")
            with open(model_eval_path, "w") as f:
                for key, value in statistics.items():
                    msg = "{}\t{}\n".format(key, _format_value(value))
                    if not cli_only:
                        f.write(msg)        
                    print(msg)

            if not cli_only:
                print("\nSaved model evaluation statistics to {}".format(model_eval_path))
            
            if run is not None:
                run.save(model_eval_path)
        
        except Exception as e:
            print(f"Summary writer error... {e}")


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

def assign_node_labels_and_colors(g):
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

    for idx, node in enumerate(g_homo.ndata[dgl.NTYPE]):
        labels[idx] = (
            node_type_order[node] + "_" + str(int(g_homo.ndata[dgl.NID][idx]))
        )
        colors.append(color_dict[node_type_order[node]])

    return labels, colors

def plot_eval_graphs(dir, graphs_to_plot, eval_it=None):
    options = {
        "node_size": 300,
        "width": 1,
        "with_labels": True,
        "font_size": 12,
        "font_color": "r",
    }

    for i, g in enumerate(graphs_to_plot):
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))
        labels, colors = assign_node_labels_and_colors(g)
        G = dgl.to_networkx(dgl.to_homogeneous(g))
        nx.draw(G, ax=ax, node_color=colors, labels=labels, **options)
        plt.savefig(dir + f"eval_{eval_it}_graph_{i}")
        plt.close()