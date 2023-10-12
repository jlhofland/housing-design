import numpy as np
import matplotlib.pyplot as plt

# Get dataset
data = np.load("./housegan_clean_data.npy", None, True)

# Get room labels
ROOM_CLASS = {
    1:"living_room", 
    2:"kitchen", 
    3:"bedroom", 
    4:"bathroom", 
    5:"missing", 
    6:"closet", 
    7:"balcony", 
    8:"corridor", 
    9:"dining_room", 
    10:"laundry_room",
    11:"exterior_wall"}

# Define house number
house_nr = 107384

# Retrieve house-specific data
room_types = data[house_nr][0]
room_bbs = data[house_nr][1]
edges = np.array(data[house_nr][2])[:,0:4]
edge_mapping = data[house_nr][3]
doors = data[house_nr][4]


# draw a graph of all the bounding boxes in the house (room_bbs)
# [x0, y0, x1, y1], where (x0, y0) and (x1, y1) are bottom-left and top-right corners
def drawGraph(room_types, room_bbs, edges, edge_mapping, edge_features, node_features, edge_types, doors):
    # Initialise the subplot function using number of rows and columns 
    figure, axis = plt.subplots(2, 2) 

    # Draw bounding boxes
    for i in range(len(room_bbs)):
        bb = room_bbs[i]
        axis[1, 1].plot([bb[0], bb[2]], [bb[1], bb[1]], 'k-')
        axis[1, 1].plot([bb[0], bb[2]], [bb[3], bb[3]], 'k-')
        axis[1, 1].plot([bb[0], bb[0]], [bb[1], bb[3]], 'k-')
        axis[1, 1].plot([bb[2], bb[2]], [bb[1], bb[3]], 'k-')
        
    # Draw edges
    for i in range(len(edges)):
        edge = edges[i]
        axis[0, 0].plot([edge[0], edge[2]], [edge[1], edge[3]], 'k-')

    # Draw doors
    # The doors are stored as a list of indices into the edge list
    for i in range(len(doors)):
        edge = edges[doors[i]]
        axis[0, 0].plot([edge[0], edge[2]], [edge[1], edge[3]], 'g-')

    # Draw dot at center of bounding box
    for i in range(len(room_bbs)):
        bb = room_bbs[i]
        # if node_feature is true, draw a red dot else draw a black dot
        if node_features[i]:
            axis[1, 0].plot([(bb[0]+bb[2])/2], [(bb[1]+bb[3])/2], 'ko')
            axis[0, 1].plot([(bb[0]+bb[2])/2], [(bb[1]+bb[3])/2], 'ko')
        else:
            axis[0, 1].plot([(bb[0]+bb[2])/2], [(bb[1]+bb[3])/2], 'bo')
            axis[1, 1].plot([(bb[0]+bb[2])/2], [(bb[1]+bb[3])/2], 'bo')
            axis[1, 1].text((bb[0]+bb[2])/2, (bb[1]+bb[3]+3)/2, ROOM_CLASS[room_types[i]], fontsize=8)

    # The edge mapping is list of indices that refer to the rooms that the edge is next to
    # Draw a line between the centers of the rooms
    # If a mapping only has one index, then the edge is a outside wall
    for i in range(len(edge_mapping)):
        mapping = edge_mapping[i]
        room1 = room_bbs[mapping[0]]
        room2 = room_bbs[mapping[1]]

        if (edge_types[i]):
            # plot an arrow from room1 to room2 with dashed lines
            axis[1, 0].arrow((room1[0]+room1[2])/2, (room1[1]+room1[3])/2, (room2[0]+room2[2])/2 - (room1[0]+room1[2])/2, (room2[1]+room2[3])/2 - (room1[1]+room1[3])/2, color='0.5', head_width=4, head_length=4, length_includes_head=True)

            # print angle halfway of the arrow in radians
            axis[1, 0].text((room1[0]+room1[2])/2 + ((room2[0]+room2[2])/2 - (room1[0]+room1[2])/2)/2, (room1[1]+room1[3])/2 + ((room2[1]+room2[3])/2 - (room1[1]+room1[3])/2)/2, str(int(edge_features[i][1])), fontsize=8, color='r')
        else:
            if edge_features[i][0]:
                color = 'g'
            else:
                color = '0.5'

            # plot an arrow from room1 to room2 with dashed lines
            axis[0, 1].arrow((room1[0]+room1[2])/2, (room1[1]+room1[3])/2, (room2[0]+room2[2])/2 - (room1[0]+room1[2])/2, (room2[1]+room2[3])/2 - (room1[1]+room1[3])/2, color=color, head_width=4, head_length=4, length_includes_head=True)

            # print angle halfway of the arrow in radians
            axis[0, 1].text((room1[0]+room1[2])/2 + ((room2[0]+room2[2])/2 - (room1[0]+room1[2])/2)/2, (room1[1]+room1[3])/2 + ((room2[1]+room2[3])/2 - (room1[1]+room1[3])/2)/2, edge_features[i][1], fontsize=8, color='r')

    # remove axis
    axis[0, 0].axis('off')
    axis[0, 0].set_title("Floorplan") 
    axis[0, 1].axis('off')
    axis[0, 1].set_title("Room and edges") 
    axis[1, 0].axis('off')
    axis[1, 0].set_title("Outerwalls edges") 
    axis[1, 1].axis('off')
    axis[1, 1].set_title("Bounding boxes") 
    plt.show()

# Create a function that filters the edges based on the edge_mappings that only have one index
def retrieveData(room_types, room_bbs, edges, edge_mapping, doors):
    rooms_amount = len(room_types)
    edge_features = []
    node_features = [0] * len(room_types)
    edge_types = [0] * len(edges)
    
    # Loop over edges
    for i in range(len(edge_mapping)):
        # check if edge contains door
        door = True if i in doors else False

        # Check if it is outer edge
        if len(edge_mapping[i]) == 1:
            # Check which direction the wall is facing
            if edges[i][0] == edges[i][2]:
                # vertical wall
                room_bbs.append([edges[i][0]-2, edges[i][1], edges[i][2]+2, edges[i][3]])
            else:
                # horizontal wall
                room_bbs.append([edges[i][0], edges[i][1]-2, edges[i][2], edges[i][3]+2])
            
            # Add wall bounding box to edge_mapping
            edge_mapping[i].append(len(room_bbs)-1)

            # Add wall to room_types
            room_types.append(11)
            node_features.append(True)

        # Calculate angle between the centers of the two rooms
        room1 = room_bbs[edge_mapping[i][0]]
        room2 = room_bbs[edge_mapping[i][1]]
        angle = np.arctan2((room2[1]+room2[3])/2 - (room1[1]+room1[3])/2, (room2[0]+room2[2])/2 - (room1[0]+room1[2])/2)

        # translate angle into directions
        directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        angle = directions[int(round(angle / (np.pi / 4))) % 8]
        
        # add edge features
        edge_features.append([door, angle])

    for i in range(len(edges)):
        for j in range(i, len(edges)):
            if i != j and edge_mapping[i][1] >= rooms_amount and edge_mapping[j][1] >= rooms_amount:
                # Edges (x0,y0,x1,y1)
                # Check if either point 0 or point 1 of edge i is the same as either point 0 or point 1 of edge j
                if (edges[i][0] == edges[j][0] and edges[i][1] == edges[j][1]) or (edges[i][0] == edges[j][2] and edges[i][1] == edges[j][3]) or (edges[i][2] == edges[j][0] and edges[i][3] == edges[j][1]) or (edges[i][2] == edges[j][2] and edges[i][3] == edges[j][3]):
                    edge_mapping.append([edge_mapping[i][1], edge_mapping[j][1]])

                    # Calculate between the two edges edges[i] and edges[j]
                    # Do this by first translating the edges to vectors
                    # Then calculate the angle between the vectors
                    vectors = []
                    vectors.append([edges[i][0] - edges[i][2], edges[i][1] - edges[i][3]])
                    vectors.append([edges[j][0] - edges[j][2], edges[j][1] - edges[j][3]])
                    angle = np.arccos(np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
                    
                    # Translate angle to degrees
                    angle = angle * 180 / np.pi

                    edge_features.append([0, angle])
                    edge_types.append(1)
    
    # Return
    return room_types, room_bbs, edges, edge_mapping, edge_features, node_features, edge_types

# Get data
room_types, room_bbs, edges, edge_mapping, edge_features, node_features, edge_types = retrieveData(room_types, room_bbs, edges, edge_mapping, doors)

# Draw graph
drawGraph(room_types, room_bbs, edges, edge_mapping, edge_features, node_features, edge_types, doors)