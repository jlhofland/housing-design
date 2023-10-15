import torch
import dgl
import os
import numpy as np

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
    exterior_walls_sequence = torch.LongTensor(layout["exterior_walls"])
    connections_corners = torch.LongTensor(layout["connections_corners"])[:, 0:4]
    corner_type_edge_features = torch.tensor(layout["connections_corners"], dtype=torch.float32)[:, 4].reshape(-1,1)
    connections_rooms = torch.LongTensor(layout["connections_rooms"])

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

def apply_partial_graph_input_completion_old(file_path):
        with open(file_path, 'r') as fr:
            room_number_data, exterior_walls_sequence, connections_corners_sequence, connections_rooms_sequence, corner_type_edge_features = parse_input_json(file_path=file_path)

        # Extract wall nodes and corner_edge features
        wall_src_nodes = connections_corners_sequence[:,1].flatten()
        wall_dest_nodes = connections_corners_sequence[:,3].flatten()
        exterior_walls_features = []
        for wall in exterior_walls_sequence:
            wall = wall.numpy()
            wall_start = wall[0:2]
            wall_end = wall[2:4] 
            wall_length = np.linalg.norm(wall_end - wall_start)
            exterior_walls_features.append(wall_length)
        exterior_walls_features = torch.tensor(exterior_walls_features, dtype=torch.float32).reshape(-1,1)
        # corner_type_edge_features = torch.tensor(connections_corners_sequence[:,4].reshape(-1,1).clone().detach(), dtype=torch.float32)
        
        # Extract room nodes and room_adjacency_edge features
        room_adjacency_type_edge_features = connections_rooms_sequence[:,4:]

        # Create DGLHeterograph from this data
        room_type_mapping = {0:"exterior_wall", 1:"living_room", 2:"kitchen", 3:"bedroom", 4:"bathroom", 5:"missing", 6:"closet", 7:"balcony", 8:"corridor", 9:"dining_room", 10:"laundry_room"}  
        graph_data = {
            # ('source node type', 'edge type', 'destination node type') : (torch.LongTensor([#A,#B]), torch.LongTensor([#C,#D]))
            # creates edges between node ID's A -> C, and B -> D. For multiple edges from the same node, must repeat its ID
            # Edges in DGLGraph are directed by default. For undirected edges, add edges for both directions.
            ('exterior_wall', 'corner_edge', 'exterior_wall') : (wall_src_nodes, wall_dest_nodes),
        }

        for connection in connections_rooms_sequence:
            # Create the key
            type_tuple = (room_type_mapping[connection[0].item()], 'room_adjacency_edge', room_type_mapping[connection[2].item()])
            # Extract integer ids
            connection_id_src = connection[1].item()
            connection_id_dest = connection[3].item()
            # Either append to existing tuple and update dict entry, or create new dict entry with new tuple
            if type_tuple in graph_data.keys():
                # pre-existing node/edge typed connection
                new_src_list = graph_data[type_tuple][0].tolist() + [connection_id_src]
                new_dest_list = graph_data[type_tuple][1].tolist() + [connection_id_dest]
                new_src_tensor = torch.LongTensor(new_src_list)
                new_dest_tensor = torch.LongTensor(new_dest_list)
                new_connection_tuple = (new_src_tensor, new_dest_tensor)
                graph_data[type_tuple] = new_connection_tuple
            else:
                # new node/edge typed connection
                graph_data[type_tuple] = (torch.LongTensor([connection_id_src]), torch.LongTensor([connection_id_dest]))

        
        # Initialize partial graph.
        g = dgl.heterograph(graph_data)

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
            etype_tuple = (room_type_mapping[connection[0].item()], 'room_adjacency_edge', room_type_mapping[connection[2].item()])
            # print(f"etype_tuple: {etype_tuple}")
            # print([connection[0].item(), connection[2].item()])
            edge_id = g.edge_ids(connection[1].item(), connection[3].item(), etype=etype_tuple)
            g.edges[etype_tuple].data['e'][edge_id] = connection[4:]

        def initializer(shape, dtype, ctx, range):
            return torch.tensor([99], dtype=dtype, device=ctx).repeat(shape)

        for ntype in g.ntypes:
            g.set_n_initializer(initializer, ntype=ntype)
        for etype in g.canonical_etypes:
            g.set_e_initializer(initializer, etype=etype)
        
        # Uncomment to examine built graph
        for key in graph_data:
                print(f"Keyed graph data: {key} : {graph_data[key]}")
        for etype in g.canonical_etypes:
            print(f"ET: {etype} :\n {g.edges[etype].data['e']}")

        return g

def apply_partial_graph_input_completion(file_path, node_hidden_size=16):
        with open(file_path, 'r') as fr:
            room_number_data, exterior_walls_sequence, connections_corners_sequence, connections_rooms_sequence, corner_type_edge_features = parse_input_json(file_path=file_path)

        # Extract wall nodes and corner_edge features
        wall_src_nodes = connections_corners_sequence[:,1].flatten()
        wall_dest_nodes = connections_corners_sequence[:,3].flatten()
        exterior_walls_features = []
        for wall in exterior_walls_sequence:
            wall = wall.numpy()
            wall_start = wall[0:2]
            wall_end = wall[2:4] 
            wall_length = np.linalg.norm(wall_end - wall_start)
            exterior_walls_features.append(wall_length)
        exterior_walls_features = torch.tensor(exterior_walls_features, dtype=torch.float32).reshape(-1,1)
        # corner_type_edge_features = torch.tensor(connections_corners_sequence[:,4].reshape(-1,1).clone().detach(), dtype=torch.float32)
        
        # Extract room nodes and room_adjacency_edge features
        room_adjacency_type_edge_features = connections_rooms_sequence[:,4:]

        # Create DGLHeterograph from this data
        room_types = ["exterior_wall", "living_room", "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room"] 
        edge_types = ["corner_edge", "room_adjacency_edge"]

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

        # Add generic GNN 'hv' features to all nodes
        for ntype in g.ntypes:
            g.nodes[ntype].data['hv'] = torch.rand(g.num_nodes(ntype), node_hidden_size)
        
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
            # print(f"etype_tuple: {etype_tuple}")
            # print([connection[0].item(), connection[2].item()])
            edge_id = g.edge_ids(connection[1].item(), connection[3].item(), etype=etype_tuple)
            g.edges[etype_tuple].data['e'][edge_id] = connection[4:]

        def initializer(shape, dtype, ctx, range):
            return torch.tensor([99], dtype=dtype, device=ctx).repeat(shape)

        for ntype in g.ntypes:
            g.set_n_initializer(initializer, ntype=ntype)
        for etype in g.canonical_etypes:
            g.set_e_initializer(initializer, etype=etype)
        
        # Uncomment to examine filled graph structure
        for c_et in g.canonical_etypes:
            if g.num_edges(c_et) > 0:
                print(f"Edge numbers: {c_et} : {g.num_edges(c_et)}")
                print(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}")
        for nt in g.ntypes:
            if g.num_nodes(nt) > 0:
                print(f"Node features: {nt} :\n {g.nodes[nt].data}")

        return g

print("Result:")
g = apply_partial_graph_input_completion(file_path=os.getcwd()+"/input.json")

# update some features
hvs = torch.empty((0,16))
for key in g.ndata['hv']:
    print(key)
    hvs = torch.cat((hvs, g.ndata['hv'][key]), dim=0)
print(hvs)

for ntype in g.ntypes:
    if g.num_nodes(ntype) > 0:
        for node_hv in g.nodes[ntype].data['hv']:
            print(node_hv)
            # print(node['hv'])

# for node in g.nodes['exterior_wall']:
#     print(node)
#     print(g.nodes['exterior_wall'].data)