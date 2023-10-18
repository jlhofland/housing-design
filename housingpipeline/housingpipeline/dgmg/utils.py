import datetime
import os
import random
import dgl
import numpy as np
import math
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init

########################################################################################################################
#                                                    configuration                                                     #
########################################################################################################################


def mkdir_p(path):
    import errno

    try:
        os.makedirs(path)
        print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Directory {} already exists.".format(path))
        else:
            raise


def date_filename(base_dir="./"):
    dt = datetime.datetime.now()
    return os.path.join(
        base_dir,
        "{}_{:02d}-{:02d}-{:02d}".format(
            dt.date(), dt.hour, dt.minute, dt.second
        ),
    )


def setup_log_dir(opts):
    log_dir = "{}".format(date_filename(opts["log_dir"]))
    mkdir_p(log_dir)
    return log_dir


def save_arg_dict(opts, filename="settings.txt"):
    def _format_value(v):
        if isinstance(v, float):
            return "{:.4f}".format(v)
        elif isinstance(v, int):
            return "{:d}".format(v)
        else:
            return "{}".format(v)

    save_path = os.path.join(opts["log_dir"], filename)
    with open(save_path, "w") as f:
        for key, value in opts.items():
            f.write("{}\t{}\n".format(key, _format_value(value)))
    print("Saved settings to {}".format(save_path))


def setup(args):
    opts = args.__dict__.copy()

    cudnn.benchmark = False
    cudnn.deterministic = True

    # Seed
    if opts["seed"] is None:
        opts["seed"] = random.randint(1, 10000)
    random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    # Dataset
    from configure import dataset_based_configure

    opts = dataset_based_configure(opts)

    assert (
        opts["path_to_dataset"] is not None
    ), "Expect path to dataset to be set."
    if not os.path.exists(opts["path_to_dataset"]):
        if opts["dataset"] == "cycles":
            raise ValueError("Cycles dataset no longer supported")
        elif opts["dataset"] == "houses":
            print("WE GOT HOUSES GIRL")
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

    # Optimization
    if opts["clip_grad"]:
        assert (
            opts["clip_grad"] is not None
        ), "Expect the gradient norm constraint to be set."

    # Log
    print("Prepare logging directory...")
    log_dir = setup_log_dir(opts)
    opts["log_dir"] = log_dir
    mkdir_p(log_dir + "/samples")

    plt.switch_backend("Agg")

    save_arg_dict(opts)
    pprint(opts)

    return opts


########################################################################################################################
#                                                         model                                                        #
########################################################################################################################

def parse_input_json(file_path):
    import json
    if type(file_path) is tuple:
        file_path = file_path[0]
    # Read in input data from JSONC file, "./input.jsonc"
    with open(file_path, 'r') as openfile:
        # Reading from json file into python dict
        layout = json.load(openfile)

    # Manually parse numerical/boolean data into tensor
    room_number_data = torch.zeros(6, dtype=torch.float32)
    room_number_data[0] = layout["number_of_living_rooms"]
    room_number_data[1] = int(layout["living_rooms_plus?"])
    room_number_data[2] = layout["number_of_bedrooms"]
    room_number_data[3] = int(layout["bedrooms_plus?"])
    room_number_data[4] = layout["number_of_bathrooms"]
    room_number_data[5] = int(layout["bathrooms_plus?"])

    # Parse walls / connections into tensors
    exterior_walls_sequence = torch.tensor(layout["exterior_walls"], dtype=torch.float32)
    connections_corners = torch.LongTensor(layout["connections_corners"])[:, 0:4]
        # Add reverse corner edges
    num_corners = connections_corners.shape[0]
    connections_corners = connections_corners.repeat(2,1)
    connections_corners[num_corners:, [1,3]] = connections_corners[num_corners:, [3,1]]
    connections_rooms = torch.LongTensor(layout["connections_rooms"])
        # Adding bi-directional corner edges (one for CW and one for CCW edges)
    corner_type_edge_features_cw = torch.tensor(layout["connections_corners"], dtype=torch.float32)[:, 4:].reshape(-1,2)
    corner_type_edge_features_ccw = corner_type_edge_features_cw.clone()
    corner_type_edge_features_ccw[:,0] = -corner_type_edge_features_ccw[:,0]
    corner_type_edge_features = torch.cat([corner_type_edge_features_cw, corner_type_edge_features_ccw], dim=0)
    corner_type_edge_features[:,1]  = torch.zeros((corner_type_edge_features.size()[0]))

    return room_number_data, exterior_walls_sequence, connections_corners, connections_rooms, corner_type_edge_features


def weights_init(m):
    """
    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def dgmg_message_weight_init(m):
    """
    This is similar as the function above where we initialize linear layers from a normal distribution with std
    1./10 as suggested by the author. This should only be used for the message passing functions, i.e. fe's in the
    paper.
    """

    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1.0 / 10)
            init.normal_(m.bias.data, std=1.0 / 10)
        else:
            raise ValueError("Expected the input to be of type nn.Linear!")

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(_weight_init)
    else:
        m.apply(_weight_init)


class Printer(object):
    def __init__(self, num_epochs, dataset_size, batch_size, writer=None):
        """Wrapper to track the learning progress.

        Parameters
        ----------
        num_epochs : int
            Number of epochs for training
        dataset_size : int
        batch_size : int
        writer : None or SummaryWriter
            If not None, tensorboard will be used to visualize learning curves.
        """
        super(Printer, self).__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_batches = math.ceil(dataset_size / batch_size)
        self.count = 0
        self.batch_count = 0
        self.writer = writer
        self._reset()

    def _reset(self):
        """Reset when an epoch is completed."""
        self.batch_loss = 0
        self.batch_prob = 0

    def _get_current_batch(self):
        """Get current batch index."""
        remainer = self.batch_count % self.num_batches
        if (remainer == 0):
            return self.num_batches
        else:
            return remainer

    def update(self, epoch, loss, prob):
        """Update learning progress.

        Parameters
        ----------
        epoch : int
        loss : float
        prob : float
        """
        self.count += 1
        self.batch_loss += loss
        self.batch_prob += prob

        if self.count % self.batch_size == 0:
            self.batch_count += 1
            if self.writer is not None:
                self.writer.add_scalar('train_log_prob', self.batch_loss, self.batch_count)
                self.writer.add_scalar('train_prob', self.batch_prob, self.batch_count)

            print('epoch {:d}/{:d}, batch {:d}/{:d}, averaged_loss {:.4f}, averaged_prob {:.4f}'.format(
                epoch, self.num_epochs, self._get_current_batch(),
                self.num_batches, self.batch_loss, self.batch_prob))
            self._reset()

def tensor_to_one_hot(tensor, num_classes):
    """
    Convert a torch tensor containing a pair of integers into one-hot encoded format.

    Args:
    - tensor (torch.Tensor): A torch tensor containing a pair of integers.
    - num_classes (int): The total number of classes or categories.

    Returns:
    - torch.Tensor: The one-hot encoded tensor.
    """

    # Initialize a zero-filled one-hot tensor
    one_hot = torch.zeros(tensor.shape[1], num_classes)

    # Set the corresponding indices to 1
    for id in range(tensor.shape[1]):
        one_hot[id, tensor[0][id]] = 1

    # Expand dims to keep indexing the same as for the original tensor
    return one_hot[None,:]