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
import shutil
import dgl

import torch
import torch.multiprocessing as mp
from housingpipeline.dgmg.houses import plot_eval_graphs, check_house
from housingpipeline.dgmg.model import DGMG
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from housingpipeline.dgmg.utils import Printer

# os.makedirs("./example_graphs", exist_ok=True)
# os.makedirs("./example_graph_plots", exist_ok=True)


def main(rank=None, model=None, opts=None, run=None, train_dataset=None, eval_dataset=None):

    torch.set_num_threads(1)
    if not os.path.exists("./model.pth") or opts["train"] or opts["gen_data"]:
        t1 = time.time()

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_single,
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_single,
        )

        from housingpipeline.dgmg.houses import HouseModelEvaluation, HousePrinting
        evaluator = HouseModelEvaluation(
            v_min=opts["min_size"], v_max=opts["max_size"], dir=opts["log_dir"]
        )
        printer = HousePrinting(
            num_epochs=opts["nepochs"],
            num_batches=opts["ds_size"] // opts["batch_size"],
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
            if rank == 0:
                print("If you want to use tensorboard, install tensorboardX with pip.")
            writer = None
        train_printer = Printer(
            opts["nepochs"], len(train_dataset), opts["batch_size"], writer
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

                if rank == 0:
                    print(
                        "#######################\nBegin Training\n#######################"
                    )
                graphs_to_plot = []
                for i, (user_input_path, init_data, data) in enumerate(train_data_loader):
                    if i%20  == 0:
                        print(f"PID {rank} - Beginning house {i}")

                    try:
                        # here, the "actions" refer to the cycle decision sequences
                        # log_prob is a negative value := sum of all decision log-probs (also negative). Represents log(p(G,pi)) I think?
                        # Not sure how the expression E_[p_data(G,pi)][log(p(G,pi))] is maximized this way (except by minimizing to zero log(p(G,pi)))

                        # update model's user-input path
                        model.user_input_path = user_input_path
                        # Update model's cond vector
                        model.conditioning_vector_module.update_conditioning_vector(
                            user_input_path)
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

                        if rank == 0:
                            train_printer.update(
                                epoch + 1, loss_averaged.item(), prob_averaged.item()
                            )

                        if rank == 0 and i%opts["batch_size"] == 0:
                            print(
                                f"PID {rank}: Finished training on house {(i)} with batch size: {opts['batch_size']}"
                            )

                        if batch_count % opts["batch_size"] == 0:
                            batch_number += 1
                            # printer.update(
                            #     epoch + 1,
                            #     {"averaged_loss": batch_loss, "averaged_prob": batch_prob},
                            # )

                            if opts["clip_grad"]:
                                clip_grad_norm_(
                                    model.parameters(), opts["clip_bound"])

                            optimizer.step()

                            if rank == 0:
                                run.log({
                                    'epoch': epoch,
                                    'batch': batch_count,
                                    'batch_loss': batch_loss,
                                    'averaged_prob': batch_prob,
                                })

                            batch_loss = 0
                            batch_prob = 0
                            optimizer.zero_grad()
                            if rank == 0:
                                torch.save(model.state_dict(
                                ), f"./checkpoints/dgmg_model_epoch_{epoch}_batch_{batch_count}.pth")
                                run.save(
                                    f"./checkpoints/dgmg_model_epoch_{epoch}_batch_{batch_count}.pth")

                        if batch_count % opts["eval_int"] == 0 and rank == 0:

                            print(f"\n Beginning evaluation number {eval_it}")
                            t3 = time.time()

                            model.eval()
                            for i in range(opts["eval_size"]):
                                (user_input_path, init_data, data) = next(
                                    iter(eval_data_loader))

                                slash_index = user_input_path.rfind("/")
                                # save_path = opts["log_dir"] + "/house_" + user_input_path[slash_index+1:-5] + f"_epoch_{epoch}_eval_{eval_it}_eval_{i}.json"
                                # print(f"save path: {save_path}")
                                # wandb.save(user_input_path, save_path)
                                print(
                                    f"Epoch: {epoch} Eval: {eval_it}, Data: {i}, user input file: {user_input_path[slash_index+1:]}")
                                # update model's user-input path
                                model.user_input_path = user_input_path
                                # Update model's cond vector
                                model.conditioning_vector_module.update_conditioning_vector(
                                    user_input_path)
                                model.conditioning_vector = model.conditioning_vector_module.conditioning_vector
                                # update cond vector inside the add-node agent
                                model.add_node_agent.conditioning_vector = model.conditioning_vector

                                evaluator.rollout_and_examine(
                                    model, opts["num_generated_samples"], epoch=epoch, eval_it=eval_it, data_it=i, run=run)
                                evaluator.write_summary(
                                    epoch, eval_it, data_it=i, run=run)
                            eval_it += 1

                            t4 = time.time()

                            print(
                                "It took {} to finish evaluation.\n".format(
                                    datetime.timedelta(seconds=t4 - t3)
                                )
                            )
                            model.train()
                            torch.save(model.state_dict(), "./model.pth")

                    except Exception as e:
                        batch_number += 1
                        batch_count += 1
                        batch_loss = 0
                        batch_prob = 0
                        optimizer.zero_grad()
                        print(
                            f"PID: {rank} - House number {i+1} raised error {e} \nSkipping this house.")

                    # lifull_num = user_input_path[user_input_path.rfind('/')+1:-5]
                    
                    # valid, results = check_house(model)
                    # print(f"House valid? {valid}. Results: {results}")
                    # graphs_to_plot.append(model.g)
                    # dgl.save_graphs("./example_graphs/dgmg_graph_"+str(i)+".bin", [model.g])
                    # with open("./example_graphs/"+f"graph_{i}_house_nr_{lifull_num}.txt", "w") as file:
                    #     file.write(user_input_path)

                # plot_eval_graphs("./example_graph_plots/", graphs_to_plot, eval_it="gt")
                # graphs_to_plot = []
                # print("timer waiting")
                # time.sleep(30)

        if rank == 0:
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
        if rank == 0:
            torch.save(model.state_dict(), "./model.pth")


    elif os.path.exists("./model.pth"):
        t1 = time.time()
        # Setup dataset and data loader
        eval_ui_path = "/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/user_inputs_final2"
        test_dataset = CustomDataset(user_input_folder=eval_ui_path, eval_only=True)
        dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=test_dataset.collate_single,
        )
        from housingpipeline.dgmg.houses import HouseModelEvaluation, HousePrinting

        evaluator = HouseModelEvaluation(
            v_min=opts["min_size"], v_max=opts["max_size"], dir=opts["log_dir"]
        )
        printer = HousePrinting(
            num_epochs=opts["nepochs"],
            num_batches=opts["ds_size"] // opts["batch_size"],
        )

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

        # clear eval folder
        if os.path.isdir(f"./eval_graphs/"):
                shutil.rmtree(f"./eval_graphs/")
        # for i, user_input_path in enumerate(dataloader):
        for i, user_input_path in enumerate(["/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline/input.json"]):
            if i == 1: break
            print(f"Evaluation {i}")
            lifull_num = user_input_path[user_input_path.rfind('/')+1:-5]
            # update model's user-input path
            model.user_input_path = user_input_path
            # Update model's cond vector
            model.conditioning_vector_module.update_conditioning_vector(
                user_input_path)
            model.conditioning_vector = model.conditioning_vector_module.conditioning_vector
            # update cond vector inside the add-node agent
            model.add_node_agent.conditioning_vector = model.conditioning_vector

            # document the ui file
            os.makedirs(f"./eval_graphs/{lifull_num}/", exist_ok=True)
            shutil.copy(user_input_path, f"./eval_graphs/{lifull_num}/")
            evaluator.rollout_and_examine(
                model, opts["num_generated_samples"], eval_it=i, run=run, lifull_num=lifull_num)
            evaluator.write_summary(eval_it=i, run=run, cli_only=False, lifull_num=lifull_num)
        t2 = time.time()
        del model.g
        print(
            "Job done. It took {} to finish evaluation with {} evaluations.".format(
                datetime.timedelta(seconds=t2 - t1),
                i,
            )
        )


if __name__ == "__main__":
    os.chdir("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg")

    parser = argparse.ArgumentParser(description="DGMG")

    import numpy as np
    # configure
    parser.add_argument("--seed", type=int, default=np.random.default_rng().integers(0,9999), help="random seed")

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
        default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/completed_graphs_final2.pickle",
        # default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/c.pickle",
    )
    parser.add_argument(
        "--path-to-initialization-dataset",
        type=str,
        default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/partial_graphs_final2.pickle",
        # default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/p.pickle",
    )
    parser.add_argument(
        "--path-to-ui-dataset",
        type=str,
        default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/user_inputs_final2/",
        # default="/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/u/",
    )
    parser.add_argument(
        "--path-to-user-input-file-inference",
        type=str,
        default="input.json",
    )

    # learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default="5e-4",
        help="optimizer learning rate",
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
        default="0.9",
        help="training split",
    )

    # evaluation interval
    parser.add_argument(
        "--eval_int",
        type=int,
        default="200",
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

    # use checkpoint "restart" model
    parser.add_argument(
        "--restart_path",
        type=str,
        default=None,
        help="specify a path to a nice model to use to jumpstart training",
    )

    # optimization
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,  # ALEX 10
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

    # Multiprocessing
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="number of processes",
    )

    args = parser.parse_args()

    from housingpipeline.dgmg.utils import setup
    from housingpipeline.dgmg.houses import CustomDataset

    opts = setup(args)

    if not opts["train"]:
        main(opts=opts)
    
    else:
        wandb.login(key="023ec30c43128f65f73c0d6ea0b0a67d361fb547")
        run = wandb.init(project='Graphs-DGMG', config=opts)


        # multiprocessing
        num_processes = opts["num_proc"]

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
        if opts["restart_path"] is not None:
            model.load_state_dict(torch.load(opts["restart_path"]))
        model.share_memory()

        # Divy up data
        train_datasets = []
        eval_datasets = []
        

        dataset = CustomDataset(
            opts["path_to_ui_dataset"], opts["path_to_initialization_dataset"], opts["path_to_dataset"])
        train_dataset = torch.utils.data.Subset(
            dataset, range(int(opts["train_split"]*len(dataset))))
        eval_dataset = torch.utils.data.Subset(dataset, range(
            int(opts["train_split"]*len(dataset)), len(dataset)))
        
        from math import floor
        split_qty_train = floor(len(train_dataset)/num_processes)
        split_qty_eval = floor(len(eval_dataset)/num_processes)
        for rank in range(num_processes):
            st = rank * split_qty_train
            ft = (rank + 1) * split_qty_train
            se = rank * split_qty_eval
            fe = (rank + 1) * split_qty_eval
            train_datasets.append(torch.utils.data.Subset(train_dataset, range(st, ft)))
            eval_datasets.append(torch.utils.data.Subset(eval_dataset, range(se, fe)))

        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=main, args=(rank, model, opts, run, train_datasets[rank], eval_datasets[rank]))
            print("dataset lengths for PID {}: {}, {}".format(rank, len(train_datasets[rank]), len(eval_datasets[rank])))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        run.finish()
    
