import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self, rel_names):
        super(Discriminator, self).__init__()
        # Convolution (e.g., graph convolution)
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(64, 256)
            for rel in rel_names}, aggregate='sum')
        
        # Convolutional Message Passing Neural Network (e.g., graph convolution layers)
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(256, 128)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(128, 64)
            for rel in rel_names}, aggregate='sum')
        
        # Downsampling (e.g., graph pooling or linear layer)
        self.downsample = nn.Linear(64, 1)
        
    def forward(self, g):
        # Convolution
        x = self.conv1(g, g.ndata['h'])
        
        # Convolutional Message Passing Neural Network
        x = self.conv2(g, x)
        x = self.conv3(g, x)
        

        # Initialize a list to store the mean tensors for each node type
        mean_tensors = []

        for k, v in x.items():            
            # Append the mean tensor to the list
            mean_tensors.append(v.mean(1))

        # Concatenate the mean tensors along axis 0 to combine them
        x = torch.cat(mean_tensors, dim=0)

        # Downsampling
        x = self.downsample(x)  # Take the mean over all nodes
        
        return x
