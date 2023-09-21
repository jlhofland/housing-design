import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch

graph_data = {
    ('bedroom', 'adjacent_to', 'exterior_wall') : (torch.tensor([0]), torch.tensor([0])),
    ('bathroom', 'adjacent_to', 'exterior_wall') : (torch.tensor([0,1,2,3]), torch.tensor([0,0,0,0]))
}
g1 = dgl.heterograph(graph_data)
# g1.edges['adjacent_to'].data['e'] = torch.tensor([1])

print(g1)
a=g1.edge_ids(u=3, v=0, etype=('bathroom', 'adjacent_to', 'exterior_wall'))
print(a)
num_etype = g1.num_edges(('bathroom', 'adjacent_to', 'exterior_wall'))
g1.edges[('bathroom', 'adjacent_to', 'exterior_wall')].data['e'] = torch.tensor([99]).repeat(num_etype,1)
g1.edges[('bathroom', 'adjacent_to', 'exterior_wall')].data['e'][a] = torch.tensor([2])

print(g1.num_edges(('bathroom', 'adjacent_to', 'exterior_wall')))
# print(g1.edges['adjacent_to'])

# g1.add_nodes(num=1, ntype='bathroom')
# g1.add_edges(u=torch.tensor([1]), v=torch.tensor([1]), data={'f':torch.tensor([22])}, etype=('bathroom', 'adjacent_to', 'exterior_wall'))

# print(g1)
# print(g1.edges['adjacent_to'])

