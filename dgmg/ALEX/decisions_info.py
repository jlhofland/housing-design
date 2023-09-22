node_types = ["living_room", "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room"]

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

"""

# For destination node id, the node must first have been created via a prior decision 3-tuple with i=0
# Please end the list of 3-tuples after a 3-tuple with i=0 has j == "stop"
