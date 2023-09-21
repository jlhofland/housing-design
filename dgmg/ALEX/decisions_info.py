"""

Home generation decision list

decisions : list
    decisions[i] is a 3-tuple (i, j, kv)
    - If i = 0, 
        j specifies either the type of node to add self.node_types[j] or termination with j = len(self.node_types)
        kv specifies the node feature vector
    - If i = 1, 
        j specifies either edge_type to add self.edge_types[j] or termination with j = len(self.edge_types)
        kv specifies the edge feature vector
    - If i = 2, 
        j specifies the destination node id for the added edge. With the formulation of DGMG, j must be created before the decision.
        kv specifies nothing and shall be an empty list

"""