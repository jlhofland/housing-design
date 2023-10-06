import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.data import DGLDataset
import os
import dgl.function as fn
import dgl.nn as dglnn
import torch.nn.functional as F

# Get graphs
folder = 'dgl_graphs'
files = os.listdir(folder)
g = dgl.load_graphs(os.path.join(folder, files[0]))[0][0]

# Print etypes
print(g.canonical_etypes)

# save the etypes and save to pickle
rel_names = g.canonical_etypes
import pickle
with open('rel_names.pkl', 'wb') as f:
    pickle.dump(rel_names, f)


