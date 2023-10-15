"""
Augmented Deep Generative Models of Graphs for Home Layout Graphs

IAAIP 2023 Team 2
"""
import argparse
import datetime
import time
import os
import torch

# DGMG*-specific imports
from housingpipeline.dgmg.houses import HouseModelEvaluation
from housingpipeline.dgmg.model import DGMG

def create_graph_from_user_input(user_input_path=None, model_path=None):
    
    dgmg_opts = define_opts()

    assert os.path.exists(user_input_path), "Cannot run pipeline without a filled out user input file. See ./examples/input_instructions.json for an example input file. Note, no comments can be in the final input.json file."
    
    assert os.path.exists(model_path), "Cannot run pipeline without a trained DGMG* model. Please train that component individually"

    t1 = time.time()
    evaluator = HouseModelEvaluation(
        v_min=dgmg_opts["min_size"], v_max=dgmg_opts["max_size"], dir=dgmg_opts["log_dir"]
    )
    # Initialize_model
    model = DGMG(
        v_max=dgmg_opts["max_size"],
        node_hidden_size=dgmg_opts["node_hidden_size"],
        num_prop_rounds=dgmg_opts["num_propagation_rounds"],
        # ALEX-TODO: may need to push this inside the AddNode/AddEdge functions..
        # zero for now, no node features
        node_features_size=dgmg_opts["node_features_size"],
        num_edge_feature_classes_list=dgmg_opts["num_edge_feature_classes_list"],
        room_types=dgmg_opts["room_types"],
        edge_types=dgmg_opts["edge_types"],
        gen_houses_dataset_only=False,
        user_input_path=user_input_path,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pipeline_graph = evaluator.generate_single_valid_graph(user_input_path=user_input_path, model=model)
    t2 = time.time()
    print(
        "It took {} to finish evaluation.".format(
            datetime.timedelta(seconds=t2 - t1)
        )
    )
    return pipeline_graph                                               


def define_opts():
    parser = argparse.ArgumentParser(description="DGMG")

    parser.add_argument("--seed", type=int, default=None, help="random seed")

    # log
    parser.add_argument(
        "--log-dir",
        default="./results",
        help="folder to save info like experiment configuration "
        "or model evaluation results",
    )

    args = parser.parse_args()
    from utils import setup

    dgmg_opts = setup(args)

    return dgmg_opts
