"""We intend to make our reproduction as close as possible to the original paper.
The configuration in the file is mostly from the description in the original paper
and will be loaded when setting up."""


def dataset_based_configure(opts):
    ds_configure = houses_configure
    
    opts = {**opts, **ds_configure}

    return opts

synthetic_dataset_configure = {
    "node_hidden_size": 32,
    "num_propagation_rounds": 4,
    "optimizer": "Adam",
    "nepochs": 10, #25
    "ds_size": 100,#0,
    "num_generated_samples": 20,#00,
}

houses_configure = {
    **synthetic_dataset_configure,
    **{
        "min_size": 3,
        "max_size": 30,
        "lr": 5e-4,
        "node_features_size": 2,
        "num_edge_feature_classes_list": 2*[max(3,9)], # It is convenient if the two feature predictor networks predict for the same number of classes
        "room_types": ["exterior_wall", "living_room", "kitchen", "bedroom", "bathroom", "missing", "closet", "balcony", "corridor", "dining_room", "laundry_room"] ,
        "edge_types": ["corner_edge", "room_adjacency_edge"],
    },
}
