# Description: Helper functions for GAN-JER
import dgl
import matplotlib.patches as mpatches

# Create color dict for node types
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

# Create shorthand dict for node types
node_type_dict = {
    "exterior_wall": "EW",
    "living_room": "LR",
    "kitchen": "KI",
    "bedroom": "BE",
    "bathroom": "BA",
    "missing": "MI",
    "closet": "CL",
    "balcony": "BA",
    "corridor": "CO",
    "dining_room": "DR",
    "laundry_room": "LA",
}
# List of rooms
rooms = ['balcony', 'bathroom', 'bedroom', 'closet', 'corridor', 'dining_room', 'exterior_wall', 'kitchen', 'laundry_room', 'living_room', 'missing']

# Change node_type_dict to a list of abbreviations in the same order as rooms
node_type_dict = [node_type_dict[room] for room in rooms]

def get_legend_elements():
    # use mpatches and color_dict to create some custom handles and labels.
    handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    return handles


# Assign node labels and colors
def assign_node_labels_and_colors(g):
    colors = []
    labels = {}

    # Get node-type order
    node_type_order = g.ntypes

    # Create node-type subgraph
    g_homo = dgl.to_homogeneous(g)

    for idx, node in enumerate(g_homo.ndata[dgl.NTYPE]):
        labels[idx] = (
            node_type_dict[node] + str(int(g_homo.ndata[dgl.NID][idx]))
        )
        colors.append(color_dict[node_type_order[node]])
    options = {

    }
    return labels, colors