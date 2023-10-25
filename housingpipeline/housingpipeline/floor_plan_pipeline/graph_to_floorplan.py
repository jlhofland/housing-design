import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable

from torchvision.utils import save_image
from housingpipeline.houseganpp.models.models import Generator
from housingpipeline.houseganpp.misc.utils import _init_input, draw_masks, draw_graph

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def _init(output_path):
    # Create output dir
    os.makedirs(output_path, exist_ok=True)

def init_model(model_path):
    # Initialize generator and discriminator
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model = model.eval()

    # Initialize variables
    if torch.cuda.is_available():
        model.cuda()
    
    return model

# run inference
def _infer(graph, model, prev_state=None):
    
    # configure input to the network
    z, given_masks_in, given_nds, given_eds, given_eds_f = _init_input(graph, prev_state)
    # run inference model
    with torch.no_grad():
        if torch.cuda.is_available():
            masks = model(z.to('cuda'), given_masks_in.to('cuda'), given_nds.to('cuda'), given_eds.to('cuda'), given_eds_f.to('cuda'))
        else:
            masks = model(z, given_masks_in, given_nds, given_eds, given_eds_f)
        masks = masks.detach().cpu().numpy()
    return masks

def create_floorplan_from_graph(graph, model_path, output_path):

    # Do init stuff
    _init(output_path)
    model = init_model(model_path)

    # draw real graph and groundtruth
    mks, nds, eds, eds_f = graph
    (nds, eds, eds_f) = (
        torch.tensor(nds),
        torch.tensor(eds),
        torch.tensor(eds_f)
    )
    masks = Variable(mks.type(Tensor))
    real_nodes = np.where(nds[:,:-2].detach().cpu()==1)[-1] 
    graph = [nds, eds, eds_f]
    true_graph_obj, graph_im = draw_graph([real_nodes, eds.numpy()])
    graph_im.save('./{}/final_pipeline_graph2.png'.format(output_path)) # save graph
    
    # add room types incrementally
    _types = sorted(list(set(real_nodes))) 
    selected_types = [_types[:k+1] for k in range(len(_types))] 
    
    # initialize layout
    state = {'masks': masks, 'fixed_nodes': []}
    # masks = _infer(graph, model, state)
    # im0 = draw_masks(masks.copy(), nds.numpy())
    # im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 

    # generate per room type
    for _iter, _types in enumerate(selected_types):
        _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
            if len(_types) > 0 else np.array([]) 
        state = {'masks': masks, 'fixed_nodes': _fixed_nds}
        masks = _infer(graph, model, state)
                
    # save final floorplans
    imk = draw_masks(masks.copy(), nds.numpy())
    imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
    save_image(imk, '{}/final_pipeline_floorplan.png'.format(output_path), nrow=1, normalize=False)