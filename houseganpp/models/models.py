import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps
import torch.nn.utils.spectral_norm as spectral_norm


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, x.shape[-1]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x

def compute_gradient_penalty(D, x, x_fake, given_y=None, given_w=None, \
                             nd_to_sample=None, data_parallel=None, \
                             ed_to_sample=None):
    # x: real masks
    # x_fake: fake masks
    # given_y: a list of nodes (node-type one-hot encoding)
    # given_w: a list of edges (src_node_id, validity (-1,1), dest_node_id)
    # tuple of indices refering nodes/edges to batch samples

    # with open("ALEX/data.txt", "w") as file:
    #     file.write("\nReal masks:\n"+"Shape:\n"+str(x.shape)+"\n"+str(x))
    #     file.write("\nFake masks:\n"+"Shape:\n"+str(x_fake.shape)+"\n"+str(x_fake))
    #     file.write("\nNodes:\n"+"Shape:\n"+str(given_y.shape)+"\n"+str(given_y))
    #     file.write("\nEdges:\n"+"Shape:\n"+str(given_w.shape)+"\n"+str(given_w))
    #     file.write("\nNode_to_samples:\n"+"Shape:\n"+str(nd_to_sample.shape)+"\n"+str(nd_to_sample))
    #     file.write("\nEdge_to_samples:\n"+str(ed_to_sample))
    #     file.write("\nData parallel..:\n"+str(data_parallel))
    indices = nd_to_sample, ed_to_sample
    # infer batch size from number of samples
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    u = torch.FloatTensor(x.shape[0], 1, 1).to(device)
    u.data.resize_(x.shape[0], 1, 1)
    u.uniform_(0, 1)
    # For some reason mixing the real and fake masks with random interpolation
    x_both = x.data*u + x_fake.data*(1-u)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    if data_parallel:
        _output = data_parallel(D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        # And then we have the discriminator score this mix??
        _output = D(x_both, given_y, given_w, nd_to_sample)
    # The code calculates the gradient of the discriminator's output with respect to the mixed data (x_both). 
    # It then computes the gradient penalty as the squared norm of these gradients minus 1, averaged over the batch. 
    # This penalty term encourages the gradients of the discriminator to have a norm close to 1, promoting smoother 
    # and more stable training.
    grad = torch.autograd.grad(outputs=_output, inputs=x_both, grad_outputs=grad_outputs, \
                               retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, batch_norm=False):
    block = []
    
    if upsample:
        if spec_norm:
            block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True)))
        else:
            block.append(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True)))
        else:        
            block.append(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True))
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(inplace=True))
    # elif "tanh":
    #     block.append(torch.nn.Tanh())
    return block

# Convolutional Message Passing
class CMP(nn.Module):
    def __init__(self, in_channels, feat_size):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky"))
        self.feat_size = feat_size
        self.expander = nn.Linear(2, 16 * feat_size ** 2)
        self.consolidator = nn.Sequential(*conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky"))
    def forward(self, feats, edges=None, edge_features=None):
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        
        # Find all edges that exist. 
        # positive edge ids
        pos_inds = torch.where(edges[:, 1] > 0)
        # positive src-node ids
        pos_v_src = edges[pos_inds[0], 0].long()
        # positive dest-node ids
        pos_v_dst = edges[pos_inds[0], 2].long()
        # positive src-node feature volumes
        pos_vecs_src = feats[pos_v_src.contiguous()]
        # TODO: 
        # bring in edge features at positive edge indices
        # convert to size feats.shape[-3] x feats.shape[-1] x feats.shape[-1]
        if True:
            edge_features = torch.rand(pos_inds[0].shape[0], 2, device=device)
            expanded_edge_features = self.expander(edge_features)
        else:    
            expanded_edge_features = self.expander(edge_features[pos_inds[0]])
        formatted_edge_features = expanded_edge_features.view(-1, 16, self.feat_size, self.feat_size)
        # concatenate with positive src-node feature volumes with dim=1
        pos_edge_and_node_features_src = torch.cat([pos_vecs_src, formatted_edge_features], dim=1)
        # pass through conv-block to bring back down to feats.shape[-3] x feats.shape[-1] x feats.shape[-1]
        # result = "gated feature volumes" for the source nodes
        edge_gated_features_src = self.consolidator(pos_edge_and_node_features_src)
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(edge_gated_features_src).to(device)
        edge_gated_pooled_v_pos = torch.scatter_add(input=pooled_v_pos, dim=0, index=pos_v_dst, src=edge_gated_features_src)
        
        # pool negative edges (edge does not exist)
        # I GUESS WE DON'T NEED TO INTEGRATE THE EDGE FEATURES WHEN THERE'S NO EDGES??
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = torch.scatter_add(pooled_v_neg, 0, neg_v_dst, neg_vecs_src)
        # update nodes features
        enc_in = torch.cat([feats, edge_gated_pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out
    
    def forward_old(self, feats, edges=None):
        # allocate memory
        dtype, device = feats.dtype, feats.device
        # edges is already this shape, so idk the purpose
        edges = edges.view(-1, 3)
        # Number of nodes (feature volumes) and edges
        V, E = feats.size(0), edges.size(0)
        # empty feature volumes to hold pooled features
        # interesting because pos and negative are the same size? Same 'V' rows... x the (16x8x8)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        
        # pool positive edges (edge does exist)
        # Recall that edges are of the form [(src, exists (1,-1), dest), ...]
        # Find all edges that exist. 
        pos_inds = torch.where(edges[:, 1] > 0)
        # Note: the following two lines of code INDICATE that the edges and message passing are both BI-DIRECTIONAL
        #   So, if there is an edge between k and l, messages from k go to l, and vice versa
        #   However, if "edges" includes edges from k to l AND from l to k (not default in HG++), then this implementation
        #   would perform message passing twice. Instead, I think we can have "edges" list edges in both directions
        #   and then concat edge features, pass through a conv block to bring the channels back down (praying for no loss of information),
        #   and then finally perform the pooling and encoding. /pray
        # extract out tensor of [all src node id's, then all dest node id's]
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        # extract out reversed tensor of [all dest node id's, then all src node id's]
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        # ensure pos_v_src node list is in contiguous memory, then extract feature volumes for correpsonding nodes in pos_v_src
        # so, first half is the src feature volumes, second half is the dest feature volumes, period.
        pos_vecs_src = feats[pos_v_src.contiguous()]
        # "pos_v_dst.view(-1, 1, 1, 1)" -> a shape (N+, 1, 1, 1) tensor of node ids, all dest then all src
        # the "expand_as(pos_vecs_src)" creates feature volumes (16x8x8), one for each node id in pos_v_dst, filled with that node id
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        # torch.scatter_add(input=, dim=, index=, src=)
        # we are going to add src[i,j,k,l] to input[index[i,j,k,l], j, k, l]
        # index[i,j,k,l] will be a number from 0 to V
        # or maybe easier to think that index[i] is a volume of indices
        # then, we add src[i] to input[index[i], j, k, l]
        # for our sakes, for now, maybe we can just assume that it is performing the aggregation (sum of feature volumes) along edges
        pooled_v_pos = torch.scatter_add(input=pooled_v_pos, dim=0, index=pos_v_dst, src=pos_vecs_src)
        
        # pool negative edges (edge does not exist)
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = torch.scatter_add(pooled_v_neg, 0, neg_v_dst, neg_vecs_src)
        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(146, 16 * self.init_size ** 2)) #146
        self.upsample_1 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_2 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_3 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.cmp_1 = CMP(in_channels=16, feat_size=8)
        self.cmp_2 = CMP(in_channels=16, feat_size=16)
        self.cmp_3 = CMP(in_channels=16, feat_size=32)
        self.cmp_4 = CMP(in_channels=16, feat_size=64)
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 1, 1, act="leaky"),
            *conv_block(256, 128, 3, 1, 1, act="leaky"),    
            *conv_block(128, 1, 3, 1, 1, act="tanh"))                                        
        # for finetuning
        self.l1_fixed = nn.Sequential(nn.Linear(1, 1 * self.init_size ** 2))
        self.enc_1 = nn.Sequential(
            *conv_block(2, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 16, 3, 2, 1, act="leaky"))
        self.enc_2 = nn.Sequential(
            *conv_block(32, 32, 3, 1, 1, act="leaky"),
            *conv_block(32, 16, 3, 1, 1, act="leaky"))   

    def forward(self, z, given_m=None, given_y=None, given_w=None, given_ed_f=None):
        # z, given_m=given_masks_in, given_y=given_nds, given_w=given_eds, given_ed_f=given_edge_features
        z = z.view(-1, 128)
        # include nodes
        y = given_y.view(-1, 18)
        z = torch.cat([z, y], 1)
        x = self.l1(z)      
        # f are the expanded node/noise vectors now in 3D
        f = x.view(-1, 16, self.init_size, self.init_size)
        # combine masks and noise vectors
        m = self.enc_1(given_m)
        # print(f"f_shape: {f.shape}")
        # print(f"m_shape: {m.shape}")
        f = torch.cat([f, m], 1)
        f = self.enc_2(f)
        # apply Conv-MPN
        # First step of Convolutional Message-Passing NN
        # Passing in a set of cat'd masks/feature volumes and the edges connecting them
        # This (*f.shape[1:]) deposits the output shape for each node representation following CMP
        # The -1 brings along the number of nodes
        x = self.cmp_1(f, given_w, given_ed_f).view(-1, *f.shape[1:])
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w, given_ed_f).view(-1, *x.shape[1:])   
        x = self.upsample_2(x)
        x = self.cmp_3(x, given_w, given_ed_f).view(-1, *x.shape[1:])   
        x = self.upsample_3(x)
        x = self.cmp_4(x, given_w, given_ed_f).view(-1, *x.shape[1:])   
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, *x.shape[2:])
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            *conv_block(9, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))
        self.l1 = nn.Sequential(nn.Linear(18, 8 * 64 ** 2))
        self.cmp_1 = CMP(in_channels=16, feat_size=64)
        self.downsample_1 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_2 = CMP(in_channels=16, feat_size=32)
        self.downsample_2 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_3 = CMP(in_channels=16, feat_size=16)
        self.downsample_3 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_4 = CMP(in_channels=16, feat_size=8)

        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 2, 1, act="leaky"),
            *conv_block(256, 128, 3, 2, 1, act="leaky"),
            *conv_block(128, 128, 3, 2, 1, act="leaky"))
        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))
        self.fc_layer_local = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None, given_ed_f=None):
        x = x.view(-1, 1, 64, 64)

        # include nodes
        y = given_y
        y = self.l1(y)
        y = y.view(-1, 8, 64, 64)
        x = torch.cat([x, y], 1)
        # message passing -- Conv-MPN
        x = self.encoder(x)
        x = self.cmp_1(x, given_w, given_ed_f).view(-1, *x.shape[1:])  
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w, given_ed_f).view(-1, *x.shape[1:])
        x = self.downsample_2(x)
        x = self.cmp_3(x, given_w, given_ed_f).view(-1, *x.shape[1:])
        x = self.downsample_3(x)
        x = self.cmp_4(x, given_w, given_ed_f).view(-1, *x.shape[1:])
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, x.shape[1])
        # global loss
        x_g = add_pool(x, nd_to_sample)
        validity_global = self.fc_layer_global(x_g)
        return validity_global
    
