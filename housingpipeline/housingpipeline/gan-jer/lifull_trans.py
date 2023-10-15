# Create a function that creates bounding boxes of the outer walls (using filter_edges)
# They should be 5 pixels thick and should be expanded either x or y direction
# This depends on the orientation of the wall
# The bounding boxes should be stored in a list
def get_outer_walls(outer_edges):
    outer_walls = []
    for i in range(len(outer_edges)):
        edge = outer_edges[i]
        if edge[0] == edge[2]:
            # vertical wall
            outer_walls.append([edge[0]-2, edge[1], edge[2]+2, edge[3]])
        else:
            # horizontal wall
            outer_walls.append([edge[0], edge[1]-2, edge[2], edge[3]+2])
    return outer_walls

# Create function that draws bounding boxes
def draw_bounding_boxes(bbs):
    for i in range(len(bbs)):
        bb = bbs[i]
        plt.plot([bb[0], bb[2]], [bb[1], bb[1]], 'k-')
        plt.plot([bb[0], bb[2]], [bb[3], bb[3]], 'k-')
        plt.plot([bb[0], bb[0]], [bb[1], bb[3]], 'k-')
        plt.plot([bb[2], bb[2]], [bb[1], bb[3]], 'k-')
    plt.show()

def draw_outer_and_graph(wall_bbs, room_bbs, edges, edge_mapping, doors):
    # Draw outer walls
    for i in range(len(wall_bbs)):
        bb = wall_bbs[i]
        plt.plot([bb[0], bb[2]], [bb[1], bb[1]], 'k-')
        plt.plot([bb[0], bb[2]], [bb[3], bb[3]], 'k-')
        plt.plot([bb[0], bb[0]], [bb[1], bb[3]], 'k-')
        plt.plot([bb[2], bb[2]], [bb[1], bb[3]], 'k-')

    # Draw inner graph
    for i in range(len(edge_mapping)):
        mapping = edge_mapping[i]
        if len(mapping) == 2:
            room1 = room_bbs[mapping[0]]
            room2 = room_bbs[mapping[1]]
            plt.plot([(room1[0]+room1[2])/2, (room2[0]+room2[2])/2], [(room1[1]+room1[3])/2, (room2[1]+room2[3])/2], 'g-')

    # Show plot
    plt.show()

draw_outer_and_graph(get_outer_walls(merge_outer_edges(filter_edges(edges, edge_mapping))), room_bbs, edges, edge_mapping, doors)

# Create a function that merges outside edges that belong to the same wall
# That is edges that are approximatly on the same line
# The edges should be return in a list, take into account that the edges are not sorted
def merge_outer_edges(edges):
    merged_edges = []
    for i in range(len(edges)):
        edge = edges[i]
        if len(merged_edges) == 0:
            merged_edges.append(edge)
        else:
            merged = False
            for j in range(len(merged_edges)):
                merged_edge = merged_edges[j]
                if edge[0] == edge[2] and merged_edge[0] == merged_edge[2]:
                    # vertical wall
                    if abs(edge[0] - merged_edge[0]) < 5:
                        # merge
                        merged_edges[j] = [min(edge[0], merged_edge[0]), min(edge[1], merged_edge[1]), max(edge[2], merged_edge[2]), max(edge[3], merged_edge[3])]
                        merged = True
                        break
                elif edge[1] == edge[3] and merged_edge[1] == merged_edge[3]:
                    # horizontal wall
                    if abs(edge[1] - merged_edge[1]) < 5:
                        # merge
                        merged_edges[j] = [min(edge[0], merged_edge[0]), min(edge[1], merged_edge[1]), max(edge[2], merged_edge[2]), max(edge[3], merged_edge[3])]
                        merged = True
                        break
            if not merged:
                merged_edges.append(edge)
    return merged_edges


