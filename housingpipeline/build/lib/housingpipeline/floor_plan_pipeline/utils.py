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
    from housingpipeline.floor_plan_pipeline.configure import dataset_based_configure
    opts = dataset_based_configure(opts)

    # Log
    print("Prepare logging directory...")
    log_dir = setup_log_dir(opts)
    opts["log_dir"] = log_dir
    mkdir_p(log_dir + "/samples")

    plt.switch_backend("Agg")

    save_arg_dict(opts)
    pprint(opts)

    return opts