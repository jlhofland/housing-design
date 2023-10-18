"""
IAAIP 2023 Team 2 - Full Pipeline Script for Home Floor Plan Generation via Heterogeneous GNNs and GANs
"""
import argparse
import dgl
import os
import pickle

from housingpipeline.floor_plan_pipeline.input_to_graph import create_graph_from_user_input
from housingpipeline.floor_plan_pipeline.graph_to_floorplan import create_floorplan_from_graph

# from .input_to_graph import create_graph_from_user_input


def main(args):
    # os.chdir("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/")

    g = create_graph_from_user_input(user_input_path=args["input_path"], model_path=args["dgmg_path"])
    # dgl.data.utils.save_graphs("./misc/sample_graph.bin", g)
    print(g)
    with open("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/misc/sample_graph_list.p", "rb") as file:
        sample_graph_list = pickle.load(file)
    create_floorplan_from_graph(graph=sample_graph_list, model_path=args["hhgpp_path"], output_path=args["output_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline")

    parser.add_argument(
        "--input-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/input.json",
        help="full path to user-input file",
    )

    parser.add_argument(
        "--output-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/pipeline_output/",
        help="full path to user-input file",
    )

    parser.add_argument(
        "--dgmg-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/pretrained_models/dgmg_state_dict.pth",
        help="full path to pretained dgmg model",
    )

    parser.add_argument(
        "--hhgpp-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/pretrained_models/exp_D_9.pth",
        help="full path to pretrained hhgpp model",
    )

    args = parser.parse_args().__dict__.copy()

    main(args)