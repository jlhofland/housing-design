"""
*********************************
******* House-GAN Dataset *******
*********************************

This dataset contains 145,811 floorplans in vector format utilized for training House-GAN.
The data is organized in a list format, where each element represents one floorplan.
For each floorplan in the dataset we have the following elements (in order):

1) List of room types: mapping between room types and their corresponding class.
ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}

2) List of room bounding boxes: each bounding box is represented by [x0, y0, x1, y1], where (x0, y0) and (x1, y1) are bottom-left and top-right coordinates, respectively.

3) List of floorplan edges: edges in the floorplan are represented by [x0, y0, x1, y1, *, *], where (x0, y0) and (x1, y1) are the edges endpoints and elements * are not being used (internally used by raster-to-vector).

4) Edge to room mapping: for each edge we assign up to 2 rooms sharing that edge, normally we have 2 rooms for internal edges and 1 room for edges in the building footprint, cases with 0 room per edge (should not happen) are likely vectorization/processing errors.

5) Doors to edges list: an element "i" in this list means that the i-th edge contains a door.

6) Vector to RGB mapping: this field contains the name of the original RGB image from LIFULL dataset.
"""

import numpy as np
import matplotlib.pyplot as plt


def swap(v):
    v[:, [0, 1, 2, 3]] = v[:, [2, 3, 0, 1]]


data = np.load("housegan_clean_data.npy", None, True)

"""
DGMG Dataset creation
"""

ROOM_CLASS = {
    1: "living_room",
    2: "kitchen",
    3: "bedroom",
    4: "bathroom",
    5: "missing",
    6: "closet",
    7: "balcony",
    8: "corridor",
    9: "dining_room",
    10: "laundry_room",
}

house_nr = 2000

# Retrieve house-specific data
# edges = np.array(data[house_nr][2])[:,0:4].copy()
edges = np.ma.array(data[house_nr][2], mask=False)[:, 0:4]
doors = np.array(data[house_nr][4])
rooms = np.array(data[house_nr][1])
room_types = np.array(data[house_nr][0])

# Find bottom-left most "source" corner to start the list
exterior_walls = []

lc_temp = np.min(
    np.concatenate(
        [
            np.linalg.norm(edges[:, 0:2], axis=1, keepdims=True),
            np.linalg.norm(edges[:, 2:4], axis=1, keepdims=True),
        ],
        axis=1,
    ),
    axis=1,
).flatten()

lowest_corners = np.flatnonzero(lc_temp == lc_temp.min())

def find_most_bottom_righty_edge():
    # for example, several edges might share the same bottom-left most corner,
    # but which one extends "exterior-most"?


exterior_walls.append()

# Now, need an alorithm to repeatedly search for outermost connecting edges
"""
for the walls that connect to latest wall (see match btw (x1,y1)_0 and (x0,y0)_1)
    find the one that forms the largest interior angle
    get v0 and candidate v1's, 
    do argmin over theta = acos(v0*v1 / |v0||v1|), good from 0 to 180 deg
"""
first_edge_startpoint = edges[exterior_walls[-1], 0:2].copy()
first_edge_endpoint = edges[exterior_walls[-1], 2:4].copy()
# mask to prevent finding the first edge during endpoint matching
edges.mask[exterior_walls[-1], 2:4] = True

candidate_edge_ids = np.hstack(
    [
        np.all(first_edge_endpoint == edges[:, 0:2], axis=1).nonzero(),
        np.all(first_edge_endpoint == edges[:, 2:4], axis=1).nonzero(),
    ],
).flatten()

# ensure that the edge "starts" where the last edge "ends"
for id in candidate_edge_ids:
    if first_edge_endpoint == edges[id, 2:4]:
        swap(edges[id])

# unmask again to allow future endpoint matching to find the first edge
edges.mask[exterior_walls[-1], 2:4] = False

print([edges[candidate_edge_ids]])

v0 = edges[exterior_walls[-1], 2:4] - edges[exterior_walls[-1], 0:2]
v1s = edges[candidate_edge_ids, 2:4] - edges[candidate_edge_ids, 0:2]

angles = np.arccos(
    np.sum(v0 * v1s, axis=1) / np.linalg.norm(v0) / np.linalg.norm(v1s, axis=1)
)

next_exterior_edge_id = candidate_edge_ids[np.argmin(angles)]

exterior_walls.append(next_exterior_edge_id)


"""
Plot Single Floorplan
"""
# Plotting
fig, ax = plt.subplots()
ax.set_title("House with walls, Red ones have doors")
for num, edge in enumerate(edges):
    x = np.array([edge[0], edge[2]])
    x_avg = np.mean(x)
    y = np.array([edge[1], edge[3]])
    y_avg = np.mean(y)
    if num in doors:
        ax.plot(x, y, "r")
        plt.scatter(x_avg, y_avg, c="#FF0000")
    else:
        ax.plot(x, y, "b")
        plt.scatter(x_avg, y_avg, c="#0000FF")
for num, room in enumerate(rooms):
    center_x = np.mean([room[0], room[2]])
    center_y = np.mean([room[1], room[3]])
    plt.text(center_x, center_y, ROOM_CLASS[room_types[num]])
    plt.scatter(center_x - 4, center_y + 3, c="#000000")
ax.set_aspect("equal")
plt.show()
