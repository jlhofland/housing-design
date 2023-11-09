"""
IAAIP 2023 Team 2 - Full Pipeline Script for Home Floor Plan Generation via Heterogeneous GNNs and GANs
"""
import argparse
import dgl
import os
import pickle
from PIL import Image
from copy import deepcopy

from housingpipeline.floor_plan_pipeline.input_to_graph import create_graph_from_user_input
from housingpipeline.floor_plan_pipeline.graph_to_floorplan import create_floorplan_from_graph
from housingpipeline.dgmg.utils import dgl_to_graphlist

# from .input_to_graph import create_graph_from_user_input


def main(args):
    os.chdir(args["dir"] + "/housingpipeline/housingpipeline/floor_plan_pipeline/")

    g = create_graph_from_user_input(user_input_path=args["input_path"], model_path=args["dgmg_path"])
    graph_lista = dgl_to_graphlist(g=g, user_input_path=args["input_path"])

    # with open("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/misc/sample_graph_list.p", "rb") as file:
    #     graph_listb = pickle.load(file)

    # g = dgl.load_graphs("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg/example_graphs/dgmg_graph_0.bin")[0][0]
    # graph_listc = dgl_to_graphlist(g=g, user_input_path="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/user_inputs_final/1929.json")
    
    graph_list = graph_lista

    with open(f"./pipeline_output/graph_data.txt", "w") as file:
        file.write(f"Graph Data:\n\n")
        for c_et in g.canonical_etypes:
            if g.num_edges(c_et) > 0:
                file.write(f"Edge numbers: {c_et} : {g.num_edges(c_et)}\n")
                file.write(f"Edge features: {c_et} :\n {g.edges[c_et].data['e']}\n")
        for nt in g.ntypes:
            if g.num_nodes(nt) > 0:
                file.write(f"Node features: {nt} :\n {g.nodes[nt].data}\n")

    # with open(f"./pipeline_output/graph_data.txt", "w") as file:
    #     file.write(f"Graph Data:\n\n")
    #     for item in graph_list:
    #         if type(item) is list:
    #             for subitem in item:
    #                 file.write(str(subitem))
    #         else:
    #             file.write(str(item))
    # graph_list = []
    # mks, nds, eds, eds_f = graph_listb
    # graph_list.append(mks)
    # mks, nds, eds, eds_f = graph_lista
    # graph_list += [nds, eds, eds_f]

    # Make a copy of the graph list
    floorplan_ok = False
    total_graph_copy = deepcopy(graph_list)

    # Loop till user is satisfied with floorplan
    while not floorplan_ok:
        # Create plan
        create_floorplan_from_graph(graph=graph_list, model_path=args["hhgpp_path"], output_path=args["output_path"])

        # Open image
        plan = Image.open(args["output_path"] + "/final_pipeline_floorplan.png")
        plan.show()

        # Check if user is satisfied with floorplan
        response = input("Is this floorplan ok? Pick from (continue/regenerate/stop): ").strip().lower()
        if response == "continue":
            floorplan_ok = True
            print("Amazing your floorplan is ready and saved in the output folder.")
        elif response == "regenerate":
            plan.close()
            graph_list = deepcopy(total_graph_copy)
            print("Alright generating a new floorplan.")
        elif response == "stop":
            print("No problem, try again next time. Byeee")
            exit()
        else:
            print("Invalid input. Please enter either 'continue', 'regenerate or 'stop'.")


if __name__ == "__main__":
    
    # os.chdir("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline")

    parser = argparse.ArgumentParser(description="Full Pipeline")

    parser.add_argument(
        "--dir",
        default="/home/evalexii/Documents/IAAIPss/housing-design/",
        help="absolute path to repo",
    )

    parser.add_argument(
        "--input-path",
        default="./input.json",
        help="full path to user-input file",
    )

    parser.add_argument(
        "--output-path",
        default="./pipeline_output/",
        help="full path to user-input file",
    )

    parser.add_argument(
        "--dgmg-path",
        default="./pretrained_models/best_sparkling_planet.pth",
        help="full path to pretained dgmg model",
    )

    parser.add_argument(
        "--hhgpp-path",
        # 123000 is best so far
        default="./pretrained_models/gen_exp_D_283000.pth",
        help="full path to pretrained hhgpp model",
    )

    args = parser.parse_args().__dict__.copy()

    main(args)