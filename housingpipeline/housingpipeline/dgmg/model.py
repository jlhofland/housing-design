from functools import partial
from housingpipeline.dgmg.utils import parse_input_json, tensor_to_one_hot
from housingpipeline.dgmg.houses import HouseDataset, generate_home_dataset, check_house
import housingpipeline.dgmg.draw_graph_help as draw_graph_help
from housingpipeline.dgmg.graph_vis_test import show_graph

import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.h0 = torch.rand(1, hidden_dim)
        self.c0 = torch.rand(1, hidden_dim)

    def forward(self, x):
        # print(f"x shape: {x.shape}, 0 el: {x.shape[0]}")
        if x.shape[0] == 0:
            return torch.full((self.h0.shape[1],), fill_value=-1)
        else:
            out, _ = self.lstm(x, (self.h0, self.c0))
            return out[-1, :]  # Take the last output of the sequence


class ConditionVec(nn.Module):
    def __init__(self, file_name):
        super(ConditionVec, self).__init__()
        import os
        import torch
        import torch.nn as nn

        self.conditioning_vector = None
        self.init_conditioning_vector(file_name)

    def init_conditioning_vector(self, file_name):
        (
            room_number_data,
            exterior_walls_sequence,
            connections_corners,
            connections_rooms,
            corner_type_edge_features,
        ) = parse_input_json(file_name)

        # Encode the wall and connection sequences with LSTMs
        # num_hidden_units refers to the number of features in the short-term memory and thus the final output vector

        # expand connections

        lstm_hidden_units = 64  # Adjust as needed
        exterior_walls_input_size = exterior_walls_sequence[0].size()[0]
        connections_corners_input_size = (
            connections_corners[0].size()[0] +
            corner_type_edge_features[0].size()[0]
        )
        connections_rooms_input_size = connections_rooms[0].size()[0]
        # Sequence encoders
        self.exterior_walls_encoder = LSTMEncoder(
            input_dim=exterior_walls_input_size, hidden_dim=lstm_hidden_units
        )
        self.connections_corners_encoder = LSTMEncoder(
            input_dim=connections_corners_input_size, hidden_dim=lstm_hidden_units
        )
        self.connections_rooms_encoder = LSTMEncoder(
            input_dim=connections_rooms_input_size, hidden_dim=lstm_hidden_units
        )

        # Encode the sequences
        # walls
        exterior_walls_encoded = self.exterior_walls_encoder(
            exterior_walls_sequence)
        # corners
        connections_corners_encoded = self.connections_corners_encoder(
            torch.cat(
                [connections_corners.type(
                    torch.float32), corner_type_edge_features],
                dim=1,
            )
        )
        # rooms
        connections_rooms_encoded = self.connections_rooms_encoder(
            connections_rooms.type(torch.float32)
        )

        # Concatenate the vectors
        self.conditioning_vector = torch.cat(
            (
                room_number_data,
                exterior_walls_encoded,
                connections_corners_encoded,
                connections_rooms_encoded,
            ),
            dim=0,
        )[None, :]

        # print(self.conditioning_vector.shape)

        # # Examine encoder structure, weights
        # for params in exterior_walls_encoder.state_dict().keys():
        #     print(params)
        #     print(exterior_walls_encoder.state_dict()[params].shape)

    def update_conditioning_vector(self, file_name):
        (
            room_number_data,
            exterior_walls_sequence,
            connections_corners,
            connections_rooms,
            corner_type_edge_features,
        ) = parse_input_json(file_name)

        # Encode the sequences
        # walls
        exterior_walls_encoded = self.exterior_walls_encoder(
            exterior_walls_sequence)
        # corners
        connections_corners_encoded = self.connections_corners_encoder(
            torch.cat(
                [connections_corners.type(
                    torch.float32), corner_type_edge_features],
                dim=1,
            )
        )
        # rooms
        connections_rooms_encoded = self.connections_rooms_encoder(
            connections_rooms.type(torch.float32)
        )

        # Concatenate the vectors
        self.conditioning_vector = torch.cat(
            (
                room_number_data,
                exterior_walls_encoded,
                connections_corners_encoded,
                connections_rooms_encoded,
            ),
            dim=0,
        )[None, :]


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1), nn.Sigmoid())
        self.node_to_graph = nn.Linear(
            node_hidden_size, self.graph_hidden_size)

    def forward(self, g):
        if g.num_nodes() == 0:
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            # OLD:
            # hvs = g.ndata["hv"]
            # NEW:
            hvs = torch.empty((0, 16))
            for key in g.ndata["hv"]:
                hvs = torch.cat((hvs, g.ndata["hv"][key]), dim=0)
            return (self.node_gating(hvs) * self.node_to_graph(hvs)).sum(
                0, keepdim=True
            )


class GraphProp(nn.Module):
    def __init__(
        self, num_prop_rounds, node_hidden_size, ntypes, etypes, edge_features_size
    ):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        self.reduce_funcs = []
        node_update_funcs = []

        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(
                nn.Linear(
                    2 * node_hidden_size + edge_features_size,
                    self.node_activation_hidden_size,
                )
            )

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size, node_hidden_size)
            )

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

        # A bit dodgy but we'll create a list of canonical edge types here
        self.canonical_edge_types = [
            ("exterior_wall", "corner_edge", "exterior_wall")]
        sub_etypes = etypes.copy()
        sub_etypes.remove("corner_edge")
        for etype in sub_etypes:
            for src_ntype in ntypes:
                for dest_ntype in ntypes:
                    self.canonical_edge_types.append(
                        (src_ntype, etype, dest_ntype))

        self.etype_mr_dicts = []
        for t in range(self.num_prop_rounds):
            self.etype_mr_dicts.append(
                {
                    etype: (self.dgmg_msg, self.reduce_funcs[t])
                    for etype in self.canonical_edge_types
                }
            )

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        # Note that "node rep for node u is still accessed via "hv" from edgeu2v's src node.."
        # #ALEX
        # print(edges.src.keys())
        return {"m": torch.cat([edges.src["hv"], edges.data["e"]], dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data["hv"]
        m = nodes.mailbox["m"]
        message = torch.cat(
            [hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {"a": node_activation}

    def forward(self, g):
        if g.num_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                # etype_mr_dict = {etype : (self.dgmg_msg, self.reduce_funcs[t]) for etype in g.etypes}
                # g.update_all(
                #     message_func=self.dgmg_msg, reduce_func=self.reduce_funcs[t]
                # )
                g.multi_update_all(
                    etype_dict=self.etype_mr_dicts[t], cross_reducer="sum"
                )
                for key in g.ndata["hv"]:
                    current_a = g.ndata["a"][key]
                    current_hv = g.ndata["hv"][key]
                    g.nodes[key].data["hv"] = self.node_update_funcs[t](
                        current_a, current_hv
                    )
                # current_a = torch.cat([g.ndata['a'][key] for key in g.ndata['a']], dim=0)
                # current_hv = torch.cat([g.ndata['hv'][key] for key in g.ndata['hv']], dim=0)
                # g.ndata["hv"] = self.node_update_funcs[t](
                #     current_a, current_hv
                # )


def bernoulli_action_log_prob(logit, action):
    """Calculate the log p of an action with respect to a Bernoulli
    distribution. Use logit rather than prob for numerical stability."""
    if action == 0:
        return F.logsigmoid(-logit)
    else:
        return F.logsigmoid(logit)


class AddNode(nn.Module):
    def __init__(
        self,
        graph_embed_func,
        node_hidden_size,
        node_features_size,
        conditioning_vector,
        ntypes,
    ):
        super(AddNode, self).__init__()

        # ALEX: Add parametric number of nodes
        n_node_types = len(ntypes)
        self.ntypes = ntypes
        self.node_features_size = node_features_size
        self.graph_op = {"embed": graph_embed_func}
        self.conditioning_vector = conditioning_vector

        self.stop = "stop"  # n_node_types
        self.add_node = nn.Linear(
            graph_embed_func.graph_hidden_size, n_node_types + 1)

        # If to add a node, initialize its hv
        # ALEX number of embeddings should be number of node types.
        self.node_type_embed = nn.Embedding(n_node_types, node_hidden_size)
        # ALEX Here is where we add *space* for node features
        # ALEX OR maybe we do not, and instead the features are added as a separate
        self.initialize_hv = nn.Linear(
            node_hidden_size
            + graph_embed_func.graph_hidden_size
            + node_features_size
            + self.conditioning_vector.shape[-1],
            node_hidden_size,
        )

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size)

    def _initialize_node_repr(self, g, action, graph_embed):
        ntype = action[0]
        # ALEX This function passes through a linear layer a node_embed_CAT_graph_embed to calculate an initial hv
        # ALEX Here is where we would add node features (node_features = action[1])
        # ALEX This is where we would add our conditioning vector, c
        hv_init = self.initialize_hv(
            torch.cat(
                [
                    self.node_type_embed(
                        torch.LongTensor([self.ntypes.index(action[0])])
                    ),
                    graph_embed,
                    # new nodes do not have node_features
                    torch.full((1, 2), fill_value=-1),
                    self.conditioning_vector,
                ],
                dim=1,
            )
        )
        g.nodes[ntype].data["hv"][-1] = hv_init
        g.nodes[ntype].data["a"][-1] = self.init_node_activation
        # g.nodes[ntype].data['hf'][-1] = action[1]

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op["embed"](g)
        # print(f"Graph_embed: {graph_embed}")
        logits = self.add_node(graph_embed)
        probs = F.softmax(logits, dim=1)

        if not self.training:
            # ALEX: Do not need to sample features as new nodes do not have features
            action = ["dummy_node_type", torch.tensor(
                self.node_features_size * [-1])]
            sample = Categorical(probs).sample().item()
            if sample < len(self.ntypes):
                action[0] = self.ntypes[sample]
            else:
                action[0] = "stop"
        stop = bool(action[0] == self.stop)

        if not stop:
            g.add_nodes(1, ntype=action[0])
            # print(f"ADDED NODE: {action[0]}_{g.num_nodes(action[0])-1} ")
            self._initialize_node_repr(g, action, graph_embed)

        if self.training:
            if action[0] == self.stop:
                sample_log_prob = F.log_softmax(logits, dim=1)[
                    :, -1].reshape(1, -1)
            else:
                sample_log_prob = F.log_softmax(logits, dim=1)[
                    :, self.ntypes.index(action[0]): self.ntypes.index(action[0]) + 1
                ]
            self.log_prob.append(sample_log_prob)

        return stop, action[0]


class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size, edge_features_size, etypes):
        super(AddEdge, self).__init__()

        # For now, only room_adjacency_edge types will be added.
        # self.num_edge_types = len(etypes)
        self.edge_features_size = edge_features_size
        # self.etypes = etypes
        self.graph_op = {"embed": graph_embed_func}
        self.add_edge = nn.Linear(
            graph_embed_func.graph_hidden_size + node_hidden_size, 1
        )

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None, src_type=None):
        graph_embed = self.graph_op["embed"](g)
        src_embed = g.nodes[src_type].data["hv"][-1].reshape(1, -1)

        logit = self.add_edge(torch.cat([graph_embed, src_embed], dim=1))
        prob = torch.sigmoid(logit)

        if not self.training:
            # ALEX-OPTIONAL_TODO: Feature sampling occurs in ChooseDest..
            action = [0, torch.tensor(self.edge_features_size * [-99])]
            action[0] = Bernoulli(prob).sample().item()
        to_add_edge = bool(action[0] == 0)
        # print(f"Action: {action} and ToAddEdge: {to_add_edge}")

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(logit, action[0])
            self.log_prob.append(sample_log_prob)

        return to_add_edge


class FeatPredict(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatPredict, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.seq(x)


class PredictFeatures(nn.Module):
    def __init__(
        self, graph_embed_hidden_size, node_hidden_size, num_edge_feature_classes_list
    ):
        super(PredictFeatures, self).__init__()

        self.num_edge_feature_classes_list = num_edge_feature_classes_list
        self.max_num_classes = max(num_edge_feature_classes_list)

        self.feature_predictors = nn.ModuleList(
            [
                FeatPredict(
                    graph_embed_hidden_size + 2 * node_hidden_size,
                    16,
                    num_edge_feature_classes,
                )
                for num_edge_feature_classes in num_edge_feature_classes_list
            ]
        )

    def forward(self, e_input):
        feature_logits = torch.cat(
            [predictor(e_input) for predictor in self.feature_predictors], dim=0
        )

        numeral_values = torch.argmax(feature_logits, dim=1).reshape(1, -1)

        return feature_logits, numeral_values


class ChooseDestAndUpdate(nn.Module):
    def __init__(
        self,
        graph_embed_func,
        graph_prop_func,
        node_hidden_size,
        num_edge_feature_classes_list,
    ):
        super(ChooseDestAndUpdate, self).__init__()

        self.graph_op = {"embed": graph_embed_func, "prop": graph_prop_func}
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

        self.feature_loss = nn.CrossEntropyLoss()

        self.predict_features = PredictFeatures(
            graph_embed_hidden_size=graph_embed_func.graph_hidden_size,
            node_hidden_size=node_hidden_size,
            num_edge_feature_classes_list=num_edge_feature_classes_list,
        )

    def _initialize_edge_repr(self, g, src_embed, dest_embed):
        e_input = torch.cat(
            [src_embed, dest_embed, self.graph_op["embed"](g)], dim=1)

        feature_logits, pred_edge_features_u2v = self.predict_features(e_input)
        pred_edge_features_v2u = copy.deepcopy(pred_edge_features_u2v)
        pred_edge_features_v2u[0][1] = (pred_edge_features_v2u[0][1] + 4) % 8

        return feature_logits, pred_edge_features_u2v, pred_edge_features_v2u

    def calc_feature_loss(self, feature_logits, true_classes):
        # feature logits: (num_edge_features, num_classes) predictions
        # true_classes: (num_edge_features) true feature classes

        # loss1 = self.feature_loss(feature_logits, true_classes)
        loss = F.log_softmax(feature_logits, dim=1)[np.arange(true_classes.shape[0]), true_classes].sum()

        return loss

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action, src_type, src_id=None):
        if src_id == None:
            src_id = g.num_nodes(src_type) - 1

        # info for action variable - recall:
        # action[0] specifies a tuple of (destination node type, destination node id) for the added edge. Dest must be created before the decision.
        # action[1] specifies the edge feature vector with size (1,2):
        # 1.0 {0: "wall with door", 1: "wall without door", 2: "no wall, no door"}
        # 1.1 (direction from src to dest is ...): [0, 1,  2, 3,  4, 5,  6, 7,  8] ==
        #                                          [E, NE, N, NW, W, SW, S, SE, undefined]

        # Create src and possible destination lists to compute "likelihood scores"
        src_embed_expand = (
            g.nodes[src_type].data["hv"][src_id].expand(g.num_nodes() - 1, -1)
        )
        possible_dests_embed = torch.empty((0, 16))
        # Create mapping from chosen "dest_id" back to real dest id/type
        mapping = [[], []]
        reference_list = []  # to determine corresponding index for ground truth

        for key in g.ndata["hv"]:
            reference_list += [(key, idx)
                               for idx in range(g.num_nodes(key))]
            mapping[0] = mapping[0] + list(range(g.num_nodes(key)))
            mapping[1] = mapping[1] + g.num_nodes(key) * [key]
            possible_dests_embed = torch.cat(
                (possible_dests_embed, g.ndata["hv"][key]), dim=0
            )
        # remove the src node information from reference_list, mapping, and possible_dests_embed
        list_index_src_node = reference_list.index((src_type, src_id))
        reference_list.pop(list_index_src_node)
        mapping[0].pop(list_index_src_node)
        mapping[1].pop(list_index_src_node)
        possible_dests_embed = torch.cat(
            (possible_dests_embed[:list_index_src_node], possible_dests_embed[list_index_src_node+1:]))

        # for key in g.ndata["hv"]:
        #     if key == src_type:
        #         reference_list += [(key, idx)
        #                            for idx in range(g.num_nodes(key) - 1)]
        #         mapping[0] = mapping[0] + list(range(g.num_nodes(key) - 1))
        #         mapping[1] = mapping[1] + (g.num_nodes(key) - 1) * [key]
        #         possible_dests_embed = torch.cat(
        #             (possible_dests_embed, g.ndata["hv"][key][0:-1]), dim=0
        #         )
        #     else:
        #         reference_list += [(key, idx)
        #                            for idx in range(g.num_nodes(key))]
        #         mapping[0] = mapping[0] + list(range(g.num_nodes(key)))
        #         mapping[1] = mapping[1] + g.num_nodes(key) * [key]
        #         possible_dests_embed = torch.cat(
        #             (possible_dests_embed, g.ndata["hv"][key]), dim=0
        #         )
        assert src_embed_expand.shape == possible_dests_embed.shape, "wrong..."
        assert len(mapping[0]) == (g.num_nodes() -
                                   1), "Should be 1 less than num_nodes"
        assert len(reference_list) == (
            g.num_nodes() - 1
        ), "Should be 1 less than num_nodes"

        dests_scores = self.choose_dest(
            torch.cat([possible_dests_embed, src_embed_expand], dim=1)
        ).view(1, -1)
        dests_probs = F.softmax(dests_scores, dim=1)

        if not self.training:
            sample = Categorical(dests_probs).sample().item()
            dest_hv = possible_dests_embed[sample]

        if self.training:
            # print(f"Action CD: {action}")
            # Extract info from action
            # Example format: (('bedroom', 0), tensor([[1, 6]]))
            dest_type = action[0][0]
            dest_id = action[0][1]
            edge_features_u2v = action[1]
            # Some effort to create the "reversed" feature for the reversed edge
            a_1_0 = action[1][0][0].item()
            a_1_1 = action[1][0][1].item()
            if 0 <= a_1_1 and a_1_1 <= 7:
                edge_features_v2u = torch.tensor(
                    [[a_1_0, (a_1_1 + 4) % 8]]
                )  # Flip direction...
            elif a_1_1 == 8:
                edge_features_v2u = action[1]
            else:
                raise ValueError(
                    "Direction feature is wrong. Should be in range(9)")
            # Determine the id of dests_scores for GT to calc loss
            gt_idx = reference_list.index((dest_type, dest_id))
            dest_hv = possible_dests_embed[gt_idx]

        # For both training and inference, predict a destination and an edge feature!
        (
            feature_logits,
            pred_edge_features_u2v,
            pred_edge_features_v2u,
        ) = self._initialize_edge_repr(
            g,
            src_embed=src_embed_expand[0].reshape(1, -1),
            dest_embed=dest_hv.reshape(1, -1),
        )

        if not self.training:
            dest_id = mapping[0][sample]
            dest_type = mapping[1][sample]
            edge_features_u2v = pred_edge_features_u2v
            edge_features_v2u = pred_edge_features_v2u

        # print(f"Src type/ID: {src_type, src_id}, Dest type/id: {dest_type, dest_id}")
        # print(f"Graph has for src: {g.nodes[src_type]}")
        # print(f"Graph has for dest: {g.nodes[dest_type]}")

        # Add in the edges finally
        if not g.has_edges_between(
            src_id, dest_id, etype=(src_type, "room_adjacency_edge", dest_type)
        ):
            if not g.has_edges_between(
                dest_id, src_id, etype=(
                    dest_type, "room_adjacency_edge", src_type)
            ):
                # For undirected graphs, we add edges for both directions
                # so that we can perform graph propagation.
                g.add_edges(
                    u=src_id,
                    v=dest_id,
                    data={"e": edge_features_u2v.to(dtype=torch.int32)},
                    etype=(src_type, "room_adjacency_edge", dest_type),
                )
                g.add_edges(
                    u=dest_id,
                    v=src_id,
                    data={"e": edge_features_v2u.to(dtype=torch.int32)},
                    etype=(dest_type, "room_adjacency_edge", src_type),
                )
                # print("ADDED TWO EDGES")

                self.graph_op["prop"](g)

        # And accumulate our losses
        if self.training:
            if dests_probs.nelement() > 1:
                self.log_prob.append(
                    F.log_softmax(dests_scores, dim=1)[:, gt_idx: gt_idx + 1]
                )
            feature_loss = self.calc_feature_loss(
                feature_logits, action[1].type(torch.LongTensor).flatten()
            ).reshape(1, -1)
            self.log_prob.append(feature_loss)


class apply_partial_graph_input_completion(nn.Module):
    def __init__(
        self,
        node_hidden_size,
        room_types,
        canonical_edge_types,
        gen_houses_dataset_only,
    ):
        super(apply_partial_graph_input_completion, self).__init__()

        self.file_Path = None
        self.node_hidden_size = node_hidden_size
        self.room_types = room_types
        self.canonical_edge_types = canonical_edge_types
        self.gen_houses_dataset_only = gen_houses_dataset_only
        self.g = None

    def define_empty_typed_graph(self, ntypes, canonical_edge_types, edge_feature_size):
        def remove_all_edges(g):
            for etype in self.canonical_edge_types:
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

        def add_dummy_features(g):
            for ntype in g.ntypes:
                num_nids = g.num_nodes(ntype)
                g.nodes[ntype].data["hv"] = torch.zeros(
                    num_nids, self.node_hidden_size, dtype=torch.float32
                )
                g.nodes[ntype].data["a"] = torch.zeros(
                    num_nids, 2 * self.node_hidden_size, dtype=torch.float32
                )
                # No node features at this time
                # g.nodes[ntype].data['hf'] = torch.zeros(num_nids, 2 * self.node_features_size, dtype=torch.float32)

            for etype in self.canonical_edge_types:
                num_eids = g.num_edges(etype)
                if etype == ('exterior_wall', 'corner_edge', 'exterior_wall'):
                    g.edges[etype].data["e"] = torch.zeros(
                        num_eids, edge_feature_size, dtype=torch.float32
                    )
                else:
                    g.edges[etype].data["e"] = torch.zeros(
                        num_eids, edge_feature_size, dtype=torch.int32
                    )

        if not os.path.isfile("./empty_house_graph/empty_house_graph.bin"):
            graph_data = {}
            for canonical_edge_type in self.canonical_edge_types:
                nids = (torch.tensor([0]), torch.tensor([0]))
                graph_data[canonical_edge_type] = nids

            g = dgl.heterograph(graph_data)
            add_dummy_features(g)
            empty_out_graph(g)
            dgl.save_graphs("./empty_house_graph/empty_house_graph.bin", g)
            return g
        else:
            return dgl.load_graphs("./empty_house_graph/empty_house_graph.bin")[0][0]

    def finish_off_partial_graph(self):
        pass

    def forward(self, file_path):
        # Retrieve input data
        self.file_path = file_path
        (
            _,
            exterior_walls_sequence,
            connections_corners_sequence,
            connections_rooms_sequence,
            corner_type_edge_features,
        ) = parse_input_json(file_path)

        # Extract wall features
        exterior_walls_input_size = exterior_walls_sequence[0].size()[0]
        if exterior_walls_input_size == 6:
            exterior_walls_features = [[], []]
            for wall in exterior_walls_sequence:
                wall = wall.numpy()
                wall_start = wall[1:3]
                wall_end = wall[3:5]
                wall_length = np.linalg.norm(wall_end - wall_start)
                exterior_walls_features[0].append(wall_length)
                exterior_walls_features[1].append(wall[-1])
            exterior_walls_features = torch.tensor(
                exterior_walls_features, dtype=torch.float32
            )
            exterior_walls_features = torch.transpose(
                exterior_walls_features, 0, 1)
        elif exterior_walls_input_size == 3:
            exterior_walls_features = exterior_walls_sequence[:, 1:]
        else:
            raise ValueError(
                "Unsupported exterior wall sequence format. Should be (x0, y0, x1, y1, D) OR (L, D)"
            )

        # Initialize empty graph with all node and edge types pre-defined
        self.g = self.define_empty_typed_graph(
            ntypes=self.room_types,
            canonical_edge_types=self.canonical_edge_types,
            # TODO-Make this smarter... len(connections_rooms_sequence[0][4:].tolist()),
            edge_feature_size=2
        )

        def initializer(shape, dtype, ctx, range):
            return torch.tensor([-1], dtype=dtype, device=ctx).repeat(shape)

        for ntype in self.g.ntypes:
            self.g.set_n_initializer(initializer, ntype=ntype)
        for etype in self.g.canonical_etypes:
            self.g.set_e_initializer(initializer, etype=etype)

        # Uncomment to show empty graph structure
        # for c_et in g.canonical_etypes:
        #     if g.num_edges(c_et) >= 0:
        #         print(f"ET: {c_et} : {g.num_edges(c_et)}")

        # Add clockwise "corner-type" edges
        for connection in connections_corners_sequence:
            src_type, dest_type = (
                self.room_types[connection[0].item()],
                self.room_types[connection[2].item()],
            )
            etype = (src_type, "corner_edge", dest_type)
            assert (
                src_type == "exterior_wall" and dest_type == "exterior_wall"
            ), "Only exterior walls use corners"
            self.g.add_edges(
                u=connection[1].item(), v=connection[3].item(), etype=etype
            )

        for connection in connections_rooms_sequence:
            # Add forward edge
            assert [
                connection[0].item() < len(self.room_types)
                and connection[2].item() < len(self.room_types)
            ], "Connection index exceed numer of room types"
            etype = (
                self.room_types[connection[0].item()],
                "room_adjacency_edge",
                self.room_types[connection[2].item()],
            )
            e_feat = connection[4:].tolist()
            self.g.add_edges(
                u=connection[1].item(),
                v=connection[3].item(),
                data={"e": torch.tensor([e_feat]).to(dtype=torch.int32)},
                etype=etype,
            )
            # Add reverse edge
            etype = (
                self.room_types[connection[2].item()],
                "room_adjacency_edge",
                self.room_types[connection[0].item()],
            )
            if 0 <= e_feat[1] and e_feat[1] <= 7:
                self.g.add_edges(
                    u=connection[3].item(),
                    v=connection[1].item(),
                    data={
                        "e": torch.tensor(
                            [[e_feat[0], (e_feat[1] + 4) % 8]]).to(dtype=torch.int32)
                    },
                    etype=etype,
                )
            elif e_feat[1] == 8:
                self.g.add_edges(
                    u=connection[3].item(),
                    v=connection[1].item(),
                    data={"e": torch.tensor([e_feat]).to(dtype=torch.int32)},
                    etype=etype,
                )
            else:
                raise ValueError(
                    "Direction feature is wrong. Should be in range(9)")

        # Add in wall-node features
        self.g.nodes["exterior_wall"].data["hf"] = exterior_walls_features

        # Add in corner edge features
        self.g.edges["corner_edge"].data["e"] = corner_type_edge_features

        # # Uncomment to examine filled graph structure
        # for c_et in g.canonical_etypes:
        #     if g.num_edges(c_et) > 0:
        #         print(f"Edge numbers: {c_et} : {g.num_edges(c_et)}")
        #         print(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}")

        if self.gen_houses_dataset_only:
            generate_home_dataset(self.g, 100)
            raise ValueError("You made enough!")

        return self.g


class DGMG(nn.Module):
    def __init__(
        self,
        v_max,
        node_hidden_size,
        node_features_size,
        num_edge_feature_classes_list,
        num_prop_rounds,
        room_types,
        edge_types,
        gen_houses_dataset_only,
        user_input_path,
    ):
        super(DGMG, self).__init__()

        # Graph configuration
        self.node_hidden_size = node_hidden_size
        self.node_features_size = node_features_size
        self.edge_features_size = len(num_edge_feature_classes_list)
        self.v_max = v_max
        self.room_types = room_types
        self.edge_types = edge_types
        self.user_input_path = user_input_path
        # self.room_type_dict = {}
        # self.room_type_dict = {}
        # for idx, rt in enumerate(room_types):
        #     self.room_type_dict[rt] = idx
        # for idx, et in enumerate(edge_types):
        #     self.room_type_dict[et] = idx

        # Graph embedding module
        self.graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation module
        self.graph_prop = GraphProp(
            num_prop_rounds,
            node_hidden_size,
            ntypes=room_types,
            etypes=self.edge_types,
            edge_features_size=self.edge_features_size,
        )

        # Graph initialization
        self.partial_graph_agent = apply_partial_graph_input_completion(
            node_hidden_size=self.node_hidden_size,
            room_types=self.room_types,
            canonical_edge_types=self.graph_prop.canonical_edge_types,
            gen_houses_dataset_only=gen_houses_dataset_only,
        )
        # Data to finalize

        # Graph conditioning vector
        self.conditioning_vector_module = ConditionVec(
            file_name=self.user_input_path)
        self.conditioning_vector = self.conditioning_vector_module.conditioning_vector

        # Actions
        self.add_node_agent = AddNode(
            self.graph_embed,
            node_hidden_size,
            node_features_size,
            self.conditioning_vector,
            ntypes=self.room_types,
        )
        self.add_edge_agent = AddEdge(
            self.graph_embed,
            node_hidden_size,
            edge_features_size=self.edge_features_size,
            etypes=self.edge_types,
        )
        self.choose_dest_agent = ChooseDestAndUpdate(
            self.graph_embed,
            self.graph_prop,
            node_hidden_size,
            num_edge_feature_classes_list,
        )
        self.add_edge_agent_finalize_partial_graph = AddEdge(
            self.graph_embed,
            node_hidden_size,
            edge_features_size=self.edge_features_size,
            etypes=self.edge_types,
        )
        self.choose_dest_agent_finalize_partial_graph = ChooseDestAndUpdate(
            self.graph_embed,
            self.graph_prop,
            node_hidden_size,
            num_edge_feature_classes_list,
        )

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        from .utils import dgmg_message_weight_init, weights_init

        self.graph_embed.apply(weights_init)
        self.graph_prop.apply(weights_init)
        self.add_node_agent.apply(weights_init)
        self.add_edge_agent.apply(weights_init)
        self.choose_dest_agent.apply(weights_init)

        self.graph_prop.message_funcs.apply(dgmg_message_weight_init)

    @property
    def action_step(self):
        old_step_count = self.step_count
        self.step_count += 1
        # print(f"ACTION STEP #: {old_step_count}")

        return old_step_count

    @property
    def init_action_step(self):
        old_step_count = self.init_step_count
        self.init_step_count += 1
        # print(f"ACTION STEP #: {old_step_count}")

        return old_step_count

    def prepare_for_train(self):
        self.step_count = 0
        self.init_step_count = 0

        self.add_node_agent.prepare_training()
        self.add_edge_agent.prepare_training()
        self.choose_dest_agent.prepare_training()
        self.add_edge_agent_finalize_partial_graph.prepare_training()
        self.choose_dest_agent_finalize_partial_graph.prepare_training()

    def add_node_and_update(self, a=None):
        """Decide if to add a new node.
        If a new node should be added, update the graph."""

        return self.add_node_agent(self.g, a)

    def add_edge_or_not(self, a=None, src_type=None, finalize_partial=False):
        """Decide if a new edge should be added."""

        if not finalize_partial:
            return self.add_edge_agent(self.g, a, src_type=src_type)
        else:
            return self.add_edge_agent_finalize_partial_graph(self.g, a, src_type=src_type)

    def choose_dest_and_update(self, a=None, src_type=None, src_id=None, finalize_partial=False):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph."""

        if not finalize_partial:
            self.choose_dest_agent(self.g, a, src_type=src_type, src_id=src_id)
        else:
            self.choose_dest_agent_finalize_partial_graph(
                self.g, a, src_type=src_type, src_id=src_id)

    def get_log_prob(self):
        an_sum = 0
        ae_sum = 0
        cd_sum = 0
        ae_fp_sum = 0
        cd_fp_sum = 0
        # print(f"log_prob: {self.add_node_agent.log_prob}")
        # print(f"log_prob: {self.add_edge_agent.log_prob}")
        # print(f"log_prob: {self.choose_dest_agent.log_prob}")
        if len(self.add_node_agent.log_prob) > 0:
            an_sum = torch.cat(self.add_node_agent.log_prob).sum()
        if len(self.add_edge_agent.log_prob) > 0:
            ae_sum = torch.cat(self.add_edge_agent.log_prob).sum()
        if len(self.choose_dest_agent.log_prob) > 0:
            cd_sum = torch.cat(self.choose_dest_agent.log_prob).sum()
        if len(self.add_edge_agent_finalize_partial_graph.log_prob) > 0:
            ae_fp_sum = torch.cat(
                self.add_edge_agent_finalize_partial_graph.log_prob).sum()
        if len(self.choose_dest_agent_finalize_partial_graph.log_prob) > 0:
            cd_fp_sum = torch.cat(
                self.choose_dest_agent_finalize_partial_graph.log_prob).sum()
        return an_sum + ae_sum + cd_sum + ae_fp_sum + cd_fp_sum

    # def init_cond_vector(self, file_path):
    #     return ConditionVec(file_name=file_path).conditioning_vector

    def forward_train(self, actions):
        while True:
            act = actions[self.action_step]
            if act[0] == 0:
                break
        # Again, "actions" = "decision sequence"
        # In order to have node/edge types and node/edge features,
        # we will use a decision sequence that is formatted as so:
        stop, last_added_node_type = self.add_node_and_update(
            a=act[1:]
        )

        while not stop:
            to_add_edge = self.add_edge_or_not(
                a=actions[self.action_step][1:], src_type=last_added_node_type
            )
            while to_add_edge:
                self.choose_dest_and_update(
                    a=actions[self.action_step][1:], src_type=last_added_node_type
                )
                to_add_edge = self.add_edge_or_not(
                    a=actions[self.action_step][1:], src_type=last_added_node_type
                )
            stop, last_added_node_type = self.add_node_and_update(
                a=actions[self.action_step][1:]
            )
        #     if stop:
        #         print("STOPPING")

        # print("################\nEND OF HOUSE\n")

        return self.get_log_prob()

    def forward_inference(self):
        stop, last_added_node_type = self.add_node_and_update()
        while (not stop) and (self.g.num_nodes() < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not(src_type=last_added_node_type)
            while to_add_edge and (num_trials < self.g.num_nodes() - 1):
                self.choose_dest_and_update(src_type=last_added_node_type)
                num_trials += 1
                to_add_edge = self.add_edge_or_not(
                    src_type=last_added_node_type)
            stop, last_added_node_type = self.add_node_and_update()

        return self.g

    def forward_pipeline(self, user_interface=True):
            # The graph we will work on
            self.g = self.partial_graph_agent(self.user_input_path)

            # Add a list of canonical_etypes for use by Graph prop...
            self.graph_prop.canonical_etypes = [
                cet for cet in self.g.canonical_etypes]

            self.initialize_partial_graph_node_representations()

            if user_interface:
                # Set default to false
                partial_ok = False

                # Loop until user is satisfied with the graph or exits
                while not partial_ok:
                    show_graph(self.g, self.user_input_path)

                    # Ask the user if the plot is correct
                    response = input(
                        "Does this graph represent your user input?: (yes/no)").strip().lower()
                    if response == 'yes':
                        partial_ok = True
                    elif response == 'no':
                        print("Exiting the program. Please make changes to user input.")
                        exit()
                    else:
                        print("Invalid input. Please enter either 'yes' or 'no'.")

            # save a copy to disk and load a new one when needed.
            dgl.save_graphs("./tmp/partial_graph.bin", self.g)

            # Set default to false
            complete_ok = False

            # Loop until user is satisfied with the graph
            while not complete_ok:

                found_one = False
                if not found_one:
                    self.forward()
                    found_one = check_house(self, quiet=False)
                    if not found_one: print("House didn't pass tests, regenerating.")
                show_graph(self.g, self.user_input_path)

                # Ask the user if the plot is correct
                response = input(
                    "What would you like to do? (continue/regenerate/stop): ").strip().lower()
                if response == 'continue':
                    complete_ok = True
                elif response == 'regenerate':
                    print("You got it!")
                    # self.g : dgl.DGLGraph = dgl.load_graphs("./tmp/partial_graph.bin")[0][0]
                elif response == 'stop':
                    print("Exiting the program.")
                    exit()
                else:
                    print(
                        "Invalid input. Please enter either 'continue', 'regenerate' or 'stop'.")

                # Close plot
                plt.close()

            # Return graph
            return self.g

    def forward(self, init_actions=None, actions=None):
        # The graph we will work on
        self.g = self.partial_graph_agent(self.user_input_path)

        # Add a list of canonical_etypes for use by Graph prop...
        self.graph_prop.canonical_etypes = [
            cet for cet in self.g.canonical_etypes]

        # Right now, the nodes do not have 'hv' or 'a' features.
        # Would like to initialize these features for all added nodes in the partial graph
        # We use the AddNode agent nn's to do this.
        self.initialize_partial_graph_node_representations()

        if self.training:
            self.prepare_for_train()
            self.finalize_partial_graph_train(init_actions)
            return self.forward_train(actions)

        else:
            # Finalize the partial graph
            self.finalize_partial_graph_inference()
            return self.forward_inference()


    def initialize_partial_graph_node_representations(self):
        for ntype in self.g.ntypes:
            if self.g.num_nodes(ntype) > 0:
                # First, initialize features with garbage to make dgl happy
                self.g.nodes[ntype].data["hv"] = torch.zeros(
                    (self.g.num_nodes(ntype), self.node_hidden_size)
                )
                self.g.nodes[ntype].data["a"] = torch.zeros(
                    (self.g.num_nodes(ntype), 2 * self.node_hidden_size)
                )
                # Then input smart values
                for i, node_hv in enumerate(self.g.nodes[ntype].data["hv"]):
                    # Node features. Exterior walls have them, the rest get [-1, -1]
                    if ntype == "exterior_wall":
                        hf = self.g.nodes[ntype].data["hf"][i].reshape(1, -1)
                    else:
                        hf = torch.full((1, 2), fill_value=-1)
                    graph_embed = self.add_node_agent.graph_op["embed"](self.g)
                    node_hv = self.add_node_agent.initialize_hv(
                        torch.cat(
                            [
                                self.add_node_agent.node_type_embed(
                                    torch.LongTensor(
                                        [self.room_types.index(ntype)])
                                ),
                                graph_embed,
                                hf,  # only exterior_walls have node_features
                                self.conditioning_vector,
                            ],
                            dim=1,
                        )
                    )
                    self.g.nodes[ntype].data["hv"][i] = node_hv

    def finalize_partial_graph_train(self, init_actions):
        # with open("./ALEX/see_order.txt", "a") as file:
        #     file.write("\n****NEW HOUSE****")
        #     for ntype in self.g.ntypes:
        #         if ntype == "exterior_wall":
        #             continue
        #         if self.g.num_nodes(ntype) > 0:
        #             for node_id in range(self.g.num_nodes(ntype)):
        #                 file.write("\n"+str(ntype)+" "+str(node_id))
        for ntype in self.g.ntypes:
            if ntype == "exterior_wall":
                continue
            if self.g.num_nodes(ntype) > 0:
                for node_id in range(self.g.num_nodes(ntype)):
                    to_add_edge = self.add_edge_or_not(
                        a=init_actions[self.init_action_step][1:],
                        src_type=ntype,
                        finalize_partial=True,
                    )
                    while to_add_edge:
                        self.choose_dest_and_update(
                            a=init_actions[self.init_action_step][1:],
                            src_type=ntype,
                            src_id=node_id,
                            finalize_partial=True,
                        )
                        to_add_edge = self.add_edge_or_not(
                            a=init_actions[self.init_action_step][1:],
                            src_type=ntype,
                            finalize_partial=True,
                        )

    def finalize_partial_graph_inference(self):
        for ntype in self.g.ntypes:
            if ntype == "exterior_wall":
                continue
            if self.g.num_nodes(ntype) > 0:
                for node_id in range(self.g.num_nodes(ntype)):
                    num_trials = 0
                    to_add_edge = self.add_edge_or_not(
                        src_type=ntype,
                        finalize_partial=True
                    )
                    while to_add_edge and (num_trials < self.g.num_nodes() - 1):
                        self.choose_dest_and_update(
                            src_type=ntype,
                            src_id=node_id,
                            finalize_partial=True
                        )
                        num_trials += 1
                        to_add_edge = self.add_edge_or_not(
                            src_type=ntype,
                            finalize_partial=True
                        )
