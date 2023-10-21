"""
Learning Deep Generative Models of Graphs
Paper: https://arxiv.org/pdf/1803.03324.pdf

This implementation works with a minibatch of size 1 only for both training and inference.
"""
import argparse
import datetime
import time
import os
import wandb

import torch
from housingpipeline.dgmg.houses import plot_and_save_graphs
from housingpipeline.dgmg.model import DGMG
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from housingpipeline.dgmg.utils import Printer

os.chdir("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg")
# os.makedirs("./example_graphs", exist_ok=True)
# os.makedirs("./example_graph_plots", exist_ok=True)

def main(opts):
    if not os.path.exists("./model.pth") or opts["train"] or opts["gen_data"]:
        t1 = time.time()

        # Setup dataset and data loader
        if opts["dataset"] == "cycles":
            raise ValueError("Cycles dataset no longer supported")
        elif opts["dataset"] == "houses":
            from housingpipeline.dgmg.houses import CustomDataset, HouseDataset, UserInputDataset, HouseModelEvaluation, HousePrinting

            dataset = CustomDataset(opts["path_to_ui_dataset"], opts["path_to_initialization_dataset"], opts["path_to_dataset"])

            train_dataset = torch.utils.data.Subset(dataset, range(int(opts["train_split"]*len(dataset))))
            eval_dataset = torch.utils.data.Subset(dataset, range(int(opts["train_split"]*len(dataset)), len(dataset)))
            # dataset = torch.utils.data.TensorDataset(UserInputDataset(fname=opts["path_to_ui_dataset"]), HouseDataset(fname=opts["path_to_initialization_dataset"]), HouseDataset(fname=opts["path_to_dataset"]))
            evaluator = HouseModelEvaluation(
                v_min=opts["min_size"], v_max=opts["max_size"], dir=opts["log_dir"]
            )
            printer = HousePrinting(
                num_epochs=opts["nepochs"],
                num_batches=opts["ds_size"] // opts["batch_size"],
            )
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_single,
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_single,
        )

        # Initialize_model
        model = DGMG(
            v_max=opts["max_size"],
            node_hidden_size=opts["node_hidden_size"],
            num_prop_rounds=opts["num_propagation_rounds"],
            node_features_size=opts["node_features_size"],
            num_edge_feature_classes_list=opts["num_edge_feature_classes_list"],
            room_types=opts["room_types"],
            edge_types=opts["edge_types"],
            gen_houses_dataset_only=opts["gen_data"],
            user_input_path="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg/input.json", 
        )
        # if torch.cuda.is_available():
        #     device = torch.device("cuda:0")
        #     model.to(device)

        # Initialize optimizer
        if opts["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=opts["lr"])
        else:
            raise ValueError("Unsupported argument for the optimizer")

        try:
            from tensorboardX import SummaryWriter

            writer = SummaryWriter(opts["log_dir"])
        except ImportError:
            print("If you want to use tensorboard, install tensorboardX with pip.")
            writer = None
        train_printer = Printer(
            opts["nepochs"], len(dataset), opts["batch_size"], writer
        )

        t2 = time.time()

        # Training
        if opts["train"]:
            model.train()
            for epoch in range(opts["nepochs"]):
                eval_it = 0
                batch_count = 0
                batch_loss = 0
                batch_prob = 0
                batch_number = 0
                optimizer.zero_grad()

                print(
                    "#######################\nBegin Training\n#######################"
                )
                print(f"Beginning batch {batch_number}")
                graphs_to_plot = []
                for i, (user_input_path, init_data, data) in enumerate(train_data_loader):
                    # here, the "actions" refer to the cycle decision sequences
                    # log_prob is a negative value := sum of all decision log-probs (also negative). Represents log(p(G,pi)) I think?
                    # Not sure how the expression E_[p_data(G,pi)][log(p(G,pi))] is maximized this way (except by minimizing to zero log(p(G,pi)))

                    # update model's user-input path
                    model.user_input_path = user_input_path
                    # Update model's cond vector
                    model.conditioning_vector_module.update_conditioning_vector(user_input_path)
                    model.conditioning_vector = model.conditioning_vector_module.conditioning_vector
                    # update cond vector inside the add-node agent
                    model.add_node_agent.conditioning_vector = model.conditioning_vector

                    if opts["gen_data"]:
                        model()

                    log_prob = model(init_actions=init_data, actions=data)
                    prob = log_prob.detach().exp()

                    loss_averaged = -log_prob / opts["batch_size"]
                    prob_averaged = prob / opts["batch_size"]

                    loss_averaged.backward(retain_graph=True)

                    batch_loss += loss_averaged.item()
                    batch_prob += prob_averaged.item()
                    batch_count += 1

                    train_printer.update(
                        epoch + 1, loss_averaged.item(), prob_averaged.item()
                    )

                    print(
                        f"Finished training on house {(i+1)} with batch size: {opts['batch_size']}"
                    )

                    if batch_count % opts["batch_size"] == 0:
                        batch_number += 1
                        # printer.update(
                        #     epoch + 1,
                        #     {"averaged_loss": batch_loss, "averaged_prob": batch_prob},
                        # )

                        if opts["clip_grad"]:
                            clip_grad_norm_(model.parameters(), opts["clip_bound"])

                        optimizer.step()

                        wandb.log({
                        'epoch': epoch,
                        'batch': batch_count,
                        'batch_loss': batch_loss,
                        'averaged_prob': batch_prob,
                        })

                        batch_loss = 0
                        batch_prob = 0
                        optimizer.zero_grad()
                        torch.save(model.state_dict(), f"./checkpoints/dgmg_model_epoch_{epoch}_batch_{batch_count}.pth")
                        wandb.save(f"./checkpoints/dgmg_model_epoch_{epoch}_batch_{batch_count}.pth")
                    
                    if batch_count % opts["eval_int"] == 0:
                        
                        print(f"\n Beginning evaluation number {eval_it}")
                        t3 = time.time()

                        model.eval()
                        for i in range(opts["eval_size"]):
                            (user_input_path, init_data, data) = next(iter(eval_data_loader))
                            
                            # slash_index = user_input_path.rfind("/")
                            # save_path = opts["log_dir"] + "/house_" + user_input_path[slash_index+1:-5] + f"_epoch_{epoch}_eval_{eval_it}_eval_{i}.json"
                            # print(f"save path: {save_path}")
                            # wandb.save(user_input_path, save_path)
                            
                            # update model's user-input path
                            model.user_input_path = user_input_path
                            # Update model's cond vector
                            model.conditioning_vector_module.update_conditioning_vector(user_input_path)
                            model.conditioning_vector = model.conditioning_vector_module.conditioning_vector
                            # update cond vector inside the add-node agent
                            model.add_node_agent.conditioning_vector = model.conditioning_vector
                            
                            evaluator.rollout_and_examine(model, opts["num_generated_samples"], epoch=epoch, eval_it=eval_it, data_it=i)
                            evaluator.write_summary(epoch, eval_it, data_it=i)
                        eval_it += 1
                        
                        t4 = time.time()

                        print(
                            "It took {} to finish evaluation.\n".format(
                                datetime.timedelta(seconds=t4 - t3)
                            )
                        )
                        model.train()
        
                    # graphs_to_plot.append(model.g)
                    # dgl.save_graphs("./example_graphs/dgmg_graph_"+str(i)+".bin", [model.g])

                # plot_and_save_graphs("./example_graph_plots/", graphs_to_plot)
                # graphs_to_plot = []

        print(
            "#######################\nTraining complete, saving last model\n#######################"
        )

        # t4 = time.time()

        # print("It took {} to setup.".format(datetime.timedelta(seconds=t2 - t1)))
        # if opts["train"]:
        #     print(
        #         "It took {} to finish training.".format(
        #             datetime.timedelta(seconds=t3 - t2)
        #         )
        #     )
        # else:
        #     print("Training skipped")
        # print(
        #     "It took {} to finish evaluation.".format(
        #         datetime.timedelta(seconds=t4 - t3)
        #     )
        # )
        # print(
        #     "--------------------------------------------------------------------------"
        # )
        # print(
        #     "On average, an epoch takes {}.".format(
        #         datetime.timedelta(seconds=(t3 - t2) / opts["nepochs"])
        #     )
        # )

        del model.g
        torch.save(model, "./model.pth")
    
        wandb.finish()

    elif os.path.exists("./model.pth"):
        t1 = time.time()
        # Setup dataset and data loader
        if opts["dataset"] == "cycles":
            raise ValueError("Cycles dataset no longer supported")

        elif opts["dataset"] == "houses":
            from housingpipeline.dgmg.houses import HouseModelEvaluation, HousePrinting

            evaluator = HouseModelEvaluation(
                v_min=opts["min_size"], v_max=opts["max_size"], dir=opts["log_dir"]
            )
            printer = HousePrinting(
                num_epochs=opts["nepochs"],
                num_batches=opts["ds_size"] // opts["batch_size"],
            )
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))
        
        model = DGMG(
            v_max=opts["max_size"],
            node_hidden_size=opts["node_hidden_size"],
            num_prop_rounds=opts["num_propagation_rounds"],
            node_features_size=opts["node_features_size"],
            num_edge_feature_classes_list=opts["num_edge_feature_classes_list"],
            room_types=opts["room_types"],
            edge_types=opts["edge_types"],
            gen_houses_dataset_only=opts["gen_data"],
            user_input_path="/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg/input.json", 
        )
        model.load_state_dict(torch.load("./model.pth"))
        model.eval()
        print(
            "#######################\nGenerating sample houses!\n#######################"
        )
        evaluator.rollout_and_examine(model, opts["num_generated_samples"])
        evaluator.write_summary()
        t2 = time.time()
        print(
            "It took {} to finish evaluation.".format(
                datetime.timedelta(seconds=t2 - t1)
            )
        )
        del model.g

    print("and here")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGMG")

    # configure
    parser.add_argument("--seed", type=int, default=9284, help="random seed")

    # dataset
    parser.add_argument(
        "--dataset",
        choices=["cycles", "houses"],
        default="houses",
        help="dataset to use",
    )
    parser.add_argument(
        "--path-to-dataset",
        type=str,
        default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/completed_graphs_reduced.p",
    )
    parser.add_argument(
        "--path-to-initialization-dataset",
        type=str,
        default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/partial_graphs_reduced.p",
    )
    parser.add_argument(
        "--path-to-ui-dataset",
        type=str,
        default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/user_inputs_new_ids/",
    )
    parser.add_argument(
        "--path-to-user-input-file-inference",
        type=str,
        default="input.json",
    )

    # train first, or just eval
    parser.add_argument(
        "-t", "--train", action="store_true", help="set True to train first"
    )

    # set flag to only generate a houses dataset
    parser.add_argument(
        "-gd",
        "--gen_data",
        action="store_true",
        help="set True to only generate a houses dataset",
    )

    # training set split size
    parser.add_argument(
        "-s",
        "--train_split",
        type=float,
        default="0.6",
        help="training split",
    )

    # evaluation interval
    parser.add_argument(
        "--eval_int",
        type=int,
        default="2",
        help="number of training houses before eval houses",
    )

    # evaluation size
    parser.add_argument(
        "--eval_size",
        type=int,
        default="1",
        help="number of eval houses",
    )

    # log
    parser.add_argument(
        "--log-dir",
        default="./results",
        help="folder to save info like experiment configuration "
        "or model evaluation results",
    )

    # optimization
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,  # ALEX 10
        help="batch size to use for training",
    )
    parser.add_argument(
        "--clip-grad",
        action="store_true",
        default=True,
        help="gradient clipping is required to prevent gradient explosion",
    )
    parser.add_argument(
        "--clip-bound",
        type=float,
        default=0.25,
        help="constraint of gradient norm for gradient clipping",
    )

    args = parser.parse_args()
    from housingpipeline.dgmg.utils import setup

    opts = setup(args)

    wandb.login(key="023ec30c43128f65f73c0d6ea0b0a67d361fb547")
    wandb.init(project='Graphs-DGMG', config=opts)

    main(opts)

