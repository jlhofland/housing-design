from functools import partial
from utils import parse_input_json, define_empty_typed_graph, apply_partial_graph_input_completion

import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
import numpy as np

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.h0 = torch.rand(1,hidden_dim)
        self.c0 = torch.rand(1,hidden_dim)

    def forward(self, x):
        out, _ = self.lstm(x, (self.h0, self.c0))
        return out[-1, :]  # Take the last output of the sequence
    

class ConditionVec(nn.Module):
    def __init__(self, file_name):
        import os
        import torch
        import torch.nn as nn

        room_number_data, exterior_walls_sequence, connections_corners, connections_rooms, corner_type_edge_features = parse_input_json(os.getcwd() + "/" + file_name)

        # Encode the wall and connection sequences with LSTMs
        # num_hidden_units refers to the number of features in the short-term memory and thus the final output vector
        lstm_hidden_units = 64  # Adjust as needed
        # Encode the sequences
        exterior_walls_encoder = LSTMEncoder(input_dim=4, hidden_dim=lstm_hidden_units)
        connections_corners_encoder = LSTMEncoder(input_dim=5, hidden_dim=lstm_hidden_units)
        connections_rooms_encoder = LSTMEncoder(input_dim=6, hidden_dim=lstm_hidden_units)

        exterior_walls_encoded = exterior_walls_encoder(exterior_walls_sequence)
        connections_corners_encoded = connections_corners_encoder(torch.cat([connections_corners.type(torch.float32), corner_type_edge_features], dim=1))
        connections_rooms_encoded = connections_rooms_encoder(connections_rooms.type(torch.float32))

        # Concatenate the vectors
        self.conditioning_vector = torch.cat((room_number_data, exterior_walls_encoded, connections_corners_encoded, connections_rooms_encoded), dim=0)[None, :]
        
        # print(self.conditioning_vector.shape)

        # # Examine encoder structure, weights
        # for params in exterior_walls_encoder.state_dict().keys():
        #     print(params)
        #     print(exterior_walls_encoder.state_dict()[params].shape)


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1), nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)

    def forward(self, g):
        if g.num_nodes() == 0:
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            hvs = g.ndata["hv"]
            return (self.node_gating(hvs) * self.node_to_graph(hvs)).sum(0, keepdim=True)


class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size):
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
                    2 * node_hidden_size + 1, self.node_activation_hidden_size
                )
            )

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size, node_hidden_size)
            )

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        # Note that "node rep for node u is still accessed via "hv" from edgeu2v's src node.."
        # #ALEX
        # print(edges.src.keys())
        return {"m": torch.cat([edges.src["hv"], edges.data["he"]], dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data["hv"]
        m = nodes.mailbox["m"]
        message = torch.cat(
            [hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2
        )
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {"a": node_activation}

    def forward(self, g):
        if g.num_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                g.update_all(
                    message_func=self.dgmg_msg, reduce_func=self.reduce_funcs[t]
                )
                g.ndata["hv"] = self.node_update_funcs[t](
                    g.ndata["a"], g.ndata["hv"]
                )


def bernoulli_action_log_prob(logit, action):
    """Calculate the log p of an action with respect to a Bernoulli
    distribution. Use logit rather than prob for numerical stability."""
    if action == 0:
        return F.logsigmoid(-logit)
    else:
        return F.logsigmoid(logit)


class AddNode(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size, node_features_size, conditioning_vector):
        super(AddNode, self).__init__()

        #ALEX: Add parametric number of nodes
        n_node_types = 6
        self.graph_op = {"embed": graph_embed_func}
        self.conditioning_vector = conditioning_vector

        self.stop = n_node_types
        self.add_node = nn.Linear(graph_embed_func.graph_hidden_size, n_node_types + 1)

        # If to add a node, initialize its hv
        #ALEX number of embeddings should be number of node types.
        self.node_type_embed = nn.Embedding(n_node_types, node_hidden_size)
        #ALEX Here is where we add *space* for node features
        #ALEX OR maybe we do not, and instead the features are added as a separate 
        self.initialize_hv = nn.Linear(
            node_hidden_size + graph_embed_func.graph_hidden_size + node_features_size + self.conditioning_vector.shape[-1],
            node_hidden_size,
        )

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size)

    def _initialize_node_repr(self, g, action, graph_embed):
        num_nodes = g.num_nodes()
        #ALEX This function passes through a linear layer a node_embed_CAT_graph_embed to calculate an initial hv
        #ALEX Here is where we would add node features
        #ALEX This is where we would add our conditioning vector, c
        print(f"Action: {action}")
        node_features = action[1]
        hv_init = self.initialize_hv(
            torch.cat(
                [
                    self.node_type_embed(torch.LongTensor([action[0]])),
                    graph_embed,
                    # node_features, #ALEX-TODO: Uncomment as needed
                    self.conditioning_vector,
                ],
                dim=1,
            )
        )
        g.nodes[num_nodes - 1].data["hv"] = hv_init
        # g.nodes[num_nodes - 1].data["hf"] = node_features #ALEX-TODO: Uncomment as needed
        g.nodes[num_nodes - 1].data["a"] = self.init_node_activation
        # #ALEX
        # print(g)
        # if num_nodes > 0: print(g.nodes[num_nodes-1])

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op["embed"](g)

        logits = self.add_node(graph_embed)
        probs = F.softmax(logits, dim=1)

        if not self.training:
            #ALEX-TODO: Need to somehow sample features.
            action = [-99, []]
            action[0] = Categorical(probs).sample().item()
        stop = bool(action[0] == self.stop)

        if not stop:
            g.add_nodes(action)
            self._initialize_node_repr(g, action, graph_embed)

        if self.training:
            sample_log_prob = F.log_softmax(logits, dim=1)[:, action: action + 1]
            self.log_prob.append(sample_log_prob)

        return stop


class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddEdge, self).__init__()

        self.num_edge_types = 3
        self.graph_op = {"embed": graph_embed_func}
        self.add_edge = nn.Linear(
            graph_embed_func.graph_hidden_size + node_hidden_size, self.num_edge_types + 1
        )

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op["embed"](g)
        src_embed = g.nodes[g.num_nodes() - 1].data["hv"]

        logits = self.add_edge(torch.cat([graph_embed, src_embed], dim=1))
        probs = F.softmax(logits, dim=1)

        if not self.training:
            action = [-99, []]
            action[0] = Categorical(probs).sample().item()
        to_add_edge = bool(action[0] < self.num_edge_types)
        print(f"Action: {action} and ToAddEdge: {to_add_edge}")

        if self.training:
            sample_log_prob = F.log_softmax(logits, dim=1)[:, action[0]: action[0] + 1]
            self.log_prob.append(sample_log_prob)

        return to_add_edge


class ChooseDestAndUpdate(nn.Module):
    def __init__(self, graph_prop_func, node_hidden_size, edge_features_size):
        super(ChooseDestAndUpdate, self).__init__()

        self.graph_op = {"prop": graph_prop_func}
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, we only add 1 to indicate its existence.
        # For multiple edge types, we can use a one hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)
        g.edges[src_list, dest_list].data["he"] = edge_repr

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, dest):
        src = g.num_nodes() - 1
        possible_dests = range(src)

        src_embed_expand = g.nodes[src].data["hv"].expand(src, -1)
        possible_dests_embed = g.nodes[possible_dests].data["hv"]

        dests_scores = self.choose_dest(
            torch.cat([possible_dests_embed, src_embed_expand], dim=1)
        ).view(1, -1)
        dests_probs = F.softmax(dests_scores, dim=1)

        if not self.training:
            dest = Categorical(dests_probs).sample().item()


        print(g)
        print(f"SRC: {src}")
        print(f"DEST: {dest}")

        if not g.has_edges_between(src, dest):
            # For undirected graphs, we add edges for both directions
            # so that we can perform graph propagation.\
            src_list = [src, dest]
            dest_list = [dest, src]

            g.add_edges(src_list, dest_list)
            self._initialize_edge_repr(g, src_list, dest_list)

            self.graph_op["prop"](g)

        if self.training:
            if dests_probs.nelement() > 1:
                self.log_prob.append(
                    F.log_softmax(dests_scores, dim=1)[:, dest : dest + 1]
                )


class DGMG(nn.Module):
    def __init__(self, v_max, node_hidden_size, node_features_size, edge_features_size, num_prop_rounds, room_types, edge_types):
        super(DGMG, self).__init__()

        # Graph configuration
        self.v_max = v_max
        self.room_types = room_types
        self.edge_types = edge_types

        # Graph conditioning vector
        self.conditioning_vector = ConditionVec("input.json").conditioning_vector

        # Graph embedding module
        self.graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation module
        self.graph_prop = GraphProp(num_prop_rounds, node_hidden_size)

        # Actions
        self.add_node_agent = AddNode(self.graph_embed, node_hidden_size, node_features_size, self.conditioning_vector)
        self.add_edge_agent = AddEdge(self.graph_embed, node_hidden_size)
        self.choose_dest_agent = ChooseDestAndUpdate(self.graph_prop, node_hidden_size, edge_features_size)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        from utils import dgmg_message_weight_init, weights_init

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
        print(f"ACTION STEP #: {old_step_count}")

        return old_step_count

    def prepare_for_train(self):
        self.step_count = 0

        self.add_node_agent.prepare_training()
        self.add_edge_agent.prepare_training()
        self.choose_dest_agent.prepare_training()

    def add_node_and_update(self, a=None):
        """Decide if to add a new node.
        If a new node should be added, update the graph."""

        return self.add_node_agent(self.g, a)

    def add_edge_or_not(self, a=None):
        """Decide if a new edge should be added."""

        return self.add_edge_agent(self.g, a)

    def choose_dest_and_update(self, a=None):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph."""

        self.choose_dest_agent(self.g, a)

    def get_log_prob(self):
        return (
            torch.cat(self.add_node_agent.log_prob).sum()
            + torch.cat(self.add_edge_agent.log_prob).sum()
            + torch.cat(self.choose_dest_agent.log_prob).sum()
        )

    def forward_train(self, actions):
        # Again, "actions" = "decision sequence"
        # In order to have node/edge types and node/edge features, 
        # we will use a decision sequence that is formatted as so:

        stop = self.add_node_and_update(a=actions[self.action_step][1:])

        while not stop:
            to_add_edge = self.add_edge_or_not(a=actions[self.action_step][1:])
            while to_add_edge:
                self.choose_dest_and_update(a=actions[self.action_step][1:])
                to_add_edge = self.add_edge_or_not(a=actions[self.action_step][1:])
            stop = self.add_node_and_update(a=actions[self.action_step][1:])

        return self.get_log_prob()

    def forward_inference(self):
        stop = self.add_node_and_update()
        while (not stop) and (self.g.num_nodes() < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not()
            while to_add_edge and (num_trials < self.g.num_nodes() - 1):
                self.choose_dest_and_update()
                num_trials += 1
                to_add_edge = self.add_edge_or_not()
            stop = self.add_node_and_update()

        return self.g

    def forward(self, actions=None):
        # The graph we will work on
        self.g = apply_partial_graph_input_completion(file_path=os.getcwd()+"/input.json", room_types=self.room_types, edge_types=self.edge_types)

        if self.training:
            self.prepare_for_train()
            return self.forward_train(actions)
        else:
            return self.forward_inference()

