import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch
import torch.nn as nn

g = dgl.load_graphs("/home/evalexii/Documents/IAAIP/housing-design/dgmg/results/2023-09-27_16-56-24/graphs/g0.bin")[0][0]

for cet in g.canonical_etypes:
    if(g.num_edges(cet)>0):
        print(cet)
nodes = []
for nt in g.ntypes:
    if g.num_nodes(nt)>0:
        print(f"NT: {nt}, num: {g.num_nodes(nt)}")
        num_nt
        nodes.append(*range(g.num(nodes(nt))))
    

# a = torch.torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(a)
# b = a.clone()
# b[1,1] = -b[1,1]
# print(f"a:\n{a}")
# print(f"b:\n{b}")

# b = b.repeat(2,1)
# print(b)
# b[3:,[0,2]] = b[3:,[2,0]]
# print(b)

# graph_data = {
#     ('bedroom', 'room_adjacency', 'exterior_wall') : (torch.torch.tensor([0]), torch.torch.tensor([0])),
#     ('bathroom', 'room_adjacency', 'exterior_wall') : (torch.torch.tensor([0]), torch.torch.tensor([0])),
#     ('exterior_wall', 'room_adjacency', 'garage') : (torch.torch.tensor([0,0]), torch.torch.tensor([0,1])),
# }

# g = dgl.heterograph(graph_data)
# print(g.nodes["exterior_wall"].data)
# for cet in g.canonical_etypes:
#     if cet[0] != "exterior_wall":
#             continue
#     for src in range(g.num_nodes(cet[0])):
#         print(f"Src ID: {src}, Out degree: {g.out_degrees(u=src, etype=cet)}, ET: {cet}")
# #         assert g.out_degrees(u=src, etype=cet) > 1, "fail"

# a = torch.torch.tensor([1,2,3])

# [print(val) for idx, val in enumerate(a)]

# b = torch.torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0.],
#                  [0., 0., 0., 0., 0., 0., 0., 1., 0.]])

# print(b[0].shape)
# print(f"am: {torch.argmax(b, dim=1).reshape(1,-1)}")

# # Example of target with class probabilities
# loss = nn.CrossEntropyLoss()
# # input = torch.randn(1, 2, requires_grad=True)
# input = torch.torch.tensor([[0,1]], dtype=torch.float32, requires_grad=True)
# target = torch.torch.tensor([0]).type(torch.float32).type(torch.Longtorch.tensor)
# print(torch.argmax(input))
# # print(input)
# # print(target)
# output = loss(input, target)
# print(output)
# output.backward()
