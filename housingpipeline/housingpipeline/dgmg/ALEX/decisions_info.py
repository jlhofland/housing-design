node_types = ['balcony', 'bathroom', 'bedroom', 'closet', 'corridor', 'dining_room', 'exterior_wall', 'kitchen', 'laundry_room', 'living_room', 'missing']
# e.g: (balcony 0, balcony 1, bathroom 0, bathroom 1, bathroom 2 
"""

Home generation decision list

decisions : list
    decisions[i] is a 3-tuple (i, j, k)
    - If i = 0, 
        j is a string that represents either the type of node to add (select from node_types variable) or to end the house generation process when j == "stop"
        k specifies nothing and is set to -1
    - If i = 1, 
        j specifies either to add an edge (j = 0) or not to add an edge (j = 1)
        k specifies nothing and is set to -1
    - If i = 2, 
        j specifies a tuple of (destination node type (string), destination node id (int)) for the added edge. Destination node must have been created before the decision via an i=0 tuple. 
        k specifies the edge feature vector of format: torch.tensor([[a, b]]) where a in range(3) and b in range(9)
        a = {0: "wall with door", 1: "wall without door", 2: "no wall, no door"}
        b = (direction from src to dest is ...): [0, 1,  2, 3,  4, 5,  6, 7,  8] == 
        //                                       [E, NE, N, NW, W, SW, S, SE, undefined] 

"""

"""
Order nodes by alphabetical name, then id , both in ascending order
If i want to add an edge from node number 3 to node (bathroom, 2)
I just do 
(1, 1, -1) - for node 0

(1, 0, -1) - for node 1
(2, ("bathroom", 2), [a, b]
(1, 1, -1)

(1, 1, -1)

(1, 1, -1)

(1, 0, -1)
(2, ("bathroom", 2), [a, b] 
(1, 1, -1)

"""

# For destination node id, the node must first have been created via a prior decision 3-tuple with i=0
# Please end the list of 3-tuples after a 3-tuple with i=0 has j == "stop"
