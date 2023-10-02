*********************************
******* House-GAN Dataset *******
*********************************

This dataset contains 145,811 floorplans in vector format utilized for training House-GAN.
The data is organized in a list format, where each element represents one floorplan. 
For each floorplan in the dataset we have the following elements (in order):

1) List of room types: mapping between room types and their corresponding class.
ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}

2) List of room bounding boxes: each bounding box is represented by [x0, y0, x1, y1], where (x0, y0) and (x1, y1) are top-left and bottom-right coordinates, respectively.

3) List of floorplan edges: edges in the floorplan are represented by [x0, y0, x1, y1, *, *], where (x0, y0) and (x1, y1) are the edges endpoints and elements * are not being used (internally used by raster-to-vector).

4) Edge to room mapping: for each edge we assign up to 2 rooms sharing that edge, normally we have 2 rooms for internal edges and 1 room for edges in the building footprint, cases with 0 room per edge (should not happen) are likely vectorization/processing errors.

5) Doors to edges list: an element "i" in this list means that the i-th edge contains a door.

6) Vector to RGB mapping: this field contains the name of the original RGB image from LIFULL dataset.