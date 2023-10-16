import dgl
import os

"""
Intent of this function is to convert a DGL-Heterograph into a list of nodes, edges, and edge features per the following format. 
This allows heterohouseganpp to then convert the graph data into a floor plan img.

    "nds": all graph nodes with features in an (Nx13) list. Each node is represented as a one-hot encoded vector with 11 classes, concatinated with the node features [length, door/no door], this is [-1,-1] for room nodes.
    "eds": all graph edges in a Ex3 list, with each edge represented as [src_node_id, +1/-1, dest_node_id] where +1 indicates an edge is present, -1 otherwise
    "eds_f": all graph edge types & features in an Nx3 list, with each entry represented as [edge type (0 - CE, or 1 - RA), edge feature 1, edge feature 2]

Steps:

    1) For nds:
        a) extract node type integers and node features for the exterior walls
        b) create [-1, -1] node features for each of the room-type nodes
        c) one-hot encode the node types
        d) concatenate the features
        e) append to list
    2) For eds:
        a) 
"""

os.chdir("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/")

g = dgl.load_graphs("./misc/sample_graph.bin")[0][0]
print(type(g))

homo_g = dgl.to_homogeneous(g, edata=['e'])

# Uncomment to print out partial graph
for c_et in g.canonical_etypes:
    if g.num_edges(c_et) > 0:
        print(f"Edge numbers: {c_et} : {g.num_edges(c_et)}")
        print(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}")
for nt in g.ntypes:
    if g.num_nodes(nt) > 0:
        print(f"Node features: {nt} :\n {g.nodes[nt].data}")