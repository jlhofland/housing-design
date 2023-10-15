"""
IAAIP 2023 Team 2 - Full Pipeline Script for Home Floor Plan Generation via Heterogeneous GNNs and GANs
"""
import argparse
from housingpipeline.floor_plan_pipeline.input_to_graph import create_graph_from_user_input

# from .input_to_graph import create_graph_from_user_input


def main(args):
    create_graph_from_user_input(user_input_path=args["input_path"], model_path=args["dgmg_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline")

    parser.add_argument(
        "--input-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/input.json",
        help="full path to user-input file",
    )

    parser.add_argument(
        "--dgmg-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/pretrained_models/dgmg_state_dict.pth",
        help="full path to pretained dgmg model",
    )

    parser.add_argument(
        "--hhgpp-path",
        default="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/pretrained_models/exp_A_122600.pth",
        help="full path to pretrained hhgpp model",
    )

    args = parser.parse_args().__dict__.copy()

    main(args)