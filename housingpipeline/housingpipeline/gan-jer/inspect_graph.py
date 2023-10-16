# Description: Inspect the graph structure of the dataset
import dgl
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
import draw_graph_help

# Get graphs
folder = 'dgl_graphs'
files = os.listdir(folder)

# Pr
i = 0
correct = False

# Loop through graphs
while not correct and i < len(files):
    # Get graph and increment counter
    g = dgl.load_graphs(os.path.join(folder, files[i]))[0][0]

    # Add legenda to plot
    plt.figure(figsize=(10, 10))
    plt.legend(handles=draw_graph_help.get_legend_elements(), loc='upper right')

    # Get labels and colors
    labels, colors = draw_graph_help.assign_node_labels_and_colors(g)

    # Translate to Homogeneous graph
    hg = dgl.to_homogeneous(g)

    # Convert to networkx
    ng = hg.to_networkx()

    # Add positional constraints for the nodes
    pos = nx.spring_layout(ng)

    # Draw the graph
    nx.draw(ng, pos=pos, node_color=colors, labels=labels, font_size=7)
    plt.show(block=False)

    # Ask the user if the plot is correct
    response = input("What would you like to do? (continue/regenerate/stop): ").strip().lower()
    if response == 'continue':
        correct = True
    elif response == 'regenerate':
        plt.close()
        i += 1
    elif response == 'stop':
        # stop the program
        print("Exiting the program.")
        exit()
    else:
        print("Invalid input. Please enter either 'continue', 'regenerate' or 'stop'.")

print("End of the program.")
