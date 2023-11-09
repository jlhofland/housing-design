# Description: Inspect the graph structure of the dataset
import dgl
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
import housingpipeline.dgmg.draw_graph_help as draw_graph_help
import json

# Get graphs
folder = 'example_graphs'
files = range(0,8)

# Pr
i = 0
correct = False

# # Loop through graphs
# while not correct and i < len(files):
#     # Get graph and increment counter
#     g : dgl.DGLGraph = dgl.load_graphs(os.path.join(folder, "dgmg_graph_" + str(files[i]) + ".bin"))[0][0]

def show_graph(g, user_input_path):
    # Add legenda to plot
    plt.figure(figsize=(25, 25))
    plt.legend(handles=draw_graph_help.get_legend_elements(), loc='best')

    # Get labels and colors
    labels, colors, walls = draw_graph_help.assign_node_labels_and_colors(g)

    # Translate to Homogeneous graph
    hg = dgl.to_homogeneous(g)

    # Convert to networkx
    ng = hg.to_networkx()

    # Add positional constraints for the nodes
    pos = nx.spring_layout(ng)

    # Open and read the JSON file
    with open(user_input_path, 'r') as file:
        data = json.load(file)

    # Should be equal
    assert len(walls) == len(data["exterior_walls"])

    # print("W", walls)

    # Get first wall
    cur = [(data["exterior_walls"][0][3]), (data["exterior_walls"][0][4])]
    end = [(data["exterior_walls"][0][1]), (data["exterior_walls"][0][2])]
    
    # Set j to 0
    j = data["exterior_walls"][0][0]

    # Create list of walls without first wall
    walls_list = data["exterior_walls"].copy()
    walls_list.pop(j)

    # While loop not closed
    while end != cur:
        pos[walls[j]] = cur
        # Set pos to end of wall
        # print(j, cur, len(walls_list))

        # Find first wall with cur as end
        for k, wall in enumerate(walls_list):
            # print("k-wall", k, wall)

            # if endpoints are equal, continue as same edge
            if (wall[3] == cur[0] and wall[4] == cur[1]):
                cur = [(wall[1]), (wall[2])]
                j = wall[0]
                walls_list.pop(k)
                break

            # If first point is equal to cur, set cur to second point
            if (wall[1] == cur[0] and wall[2] == cur[1]):
                cur = [(wall[3]), (wall[4])]
                j = wall[0]
                walls_list.pop(k)
                break

    # Set last wall
    pos[walls[j]] = end

    # Calculate center of all the positions of the walls nodes
    center = [0, 0]
    for wall in walls:
        center[0] += pos[wall][0]
        center[1] += pos[wall][1]

    # Calculate the average
    center[0] /= len(walls)
    center[1] /= len(walls)

    # Add positional constraints for the nodes
    pos = nx.spring_layout(G=ng, pos=pos, fixed=walls, k=20, iterations=100, scale=2, center=center)

    # Draw the graph
    nx.draw(ng, pos=pos, node_color=colors, labels=labels, font_size=7)
    plt.show(block=False)
    plt.savefig(fname="./pipeline_output/dgl_graph.png")

# print("End of the program.")
