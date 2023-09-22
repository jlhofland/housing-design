"""We intend to make our reproduction as close as possible to the original paper.
The configuration in the file is mostly from the description in the original paper
and will be loaded when setting up."""


def dataset_based_configure(opts):
    if opts["dataset"] == "cycles":
        ds_configure = cycles_configure
    elif opts["dataset"] == "houses":
        ds_configure = houses_configure
    else:
        raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

    opts = {**opts, **ds_configure}

    return opts


synthetic_dataset_configure = {
    "node_hidden_size": 16,
    "num_propagation_rounds": 2,
    "optimizer": "Adam",
    "nepochs": 1, #25
    "ds_size": 100,#0,
    "num_generated_samples": 100,#00,
}

cycles_configure = {
    **synthetic_dataset_configure,
    **{
        "min_size": 10,
        "max_size": 20,
        "lr": 5e-4,
    },
}

houses_configure = {
    **synthetic_dataset_configure,
    **{
        "min_size": 3,
        "max_size": 30,
        "lr": 5e-4,
        "node_features_size": 0,
        "edge_features_size": 2,
        "room_types": ["exterior_wall", "living_room", "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room"] ,
        "edge_types": ["corner_edge", "room_adjacency_edge"],
    },
}
