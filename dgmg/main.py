"""
Learning Deep Generative Models of Graphs
Paper: https://arxiv.org/pdf/1803.03324.pdf

This implementation works with a minibatch of size 1 only for both training and inference.
"""
import argparse
import datetime
import time
import os

import torch
from model import DGMG
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import Printer


def main(opts):
    if not os.path.exists("./model.pth") or opts["train"] or opts["gen_data"]:
        t1 = time.time()

        # Initialize_model
        model = DGMG(
            v_max=opts["max_size"],
            node_hidden_size=opts["node_hidden_size"],
            num_prop_rounds=opts["num_propagation_rounds"],
            # ALEX-TODO: may need to push this inside the AddNode/AddEdge functions..
            # zero for now, no node features
            node_features_size=opts["node_features_size"],
            num_edge_feature_classes_list=opts["num_edge_feature_classes_list"],
            room_types=opts["room_types"],
            edge_types=opts["edge_types"],
            gen_houses_dataset_only=opts["gen_data"],
        )
        if opts["gen_data"]:
            model()

        # Setup dataset and data loader
        if opts["dataset"] == "cycles":
            raise ValueError("Cycles dataset no longer supported")
        elif opts["dataset"] == "houses":
            from houses import HouseDataset, HouseModelEvaluation, HousePrinting

            dataset = HouseDataset(fname=opts["path_to_dataset"])
            evaluator = HouseModelEvaluation(
                v_min=opts["min_size"], v_max=opts["max_size"], dir=opts["log_dir"]
            )
            printer = HousePrinting(
                num_epochs=opts["nepochs"],
                num_batches=opts["ds_size"] // opts["batch_size"],
            )
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_single,
        )
        # model = model.cuda()

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
                batch_count = 0
                batch_loss = 0
                batch_prob = 0
                batch_number = 0
                optimizer.zero_grad()

                print(
                    "#######################\nBegin Training\n#######################"
                )
                print(f"Beginning batch {batch_number}")
                for i, data in enumerate(data_loader):
                    # here, the "actions" refer to the cycle decision sequences
                    # log_prob is a negative value := sum of all decision log-probs (also negative). Represents log(p(G,pi)) I think?
                    # Not sure how the expression E_[p_data(G,pi)][log(p(G,pi))] is maximized this way (except by minimizing to zero log(p(G,pi)))

                    log_prob = model(actions=data)
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

                        batch_loss = 0
                        batch_prob = 0
                        optimizer.zero_grad()

        t3 = time.time()

        model.eval()
        print(
            "#######################\nTraining complete, begin evaluation\n#######################"
        )
        evaluator.rollout_and_examine(model, opts["num_generated_samples"])
        evaluator.write_summary()

        t4 = time.time()

        print("It took {} to setup.".format(datetime.timedelta(seconds=t2 - t1)))
        if opts["train"]:
            print(
                "It took {} to finish training.".format(
                    datetime.timedelta(seconds=t3 - t2)
                )
            )
        else:
            print("Training skipped")
        print(
            "It took {} to finish evaluation.".format(
                datetime.timedelta(seconds=t4 - t3)
            )
        )
        print(
            "--------------------------------------------------------------------------"
        )
        print(
            "On average, an epoch takes {}.".format(
                datetime.timedelta(seconds=(t3 - t2) / opts["nepochs"])
            )
        )

        del model.g
        torch.save(model, "./model.pth")

    elif os.path.exists("./model.pth"):
        t1 = time.time()
        # Setup dataset and data loader
        if opts["dataset"] == "cycles":
            raise ValueError("Cycles dataset no longer supported")

        elif opts["dataset"] == "houses":
            from houses import HouseModelEvaluation, HousePrinting

            evaluator = HouseModelEvaluation(
                v_min=opts["min_size"], v_max=opts["max_size"], dir=opts["log_dir"]
            )
            printer = HousePrinting(
                num_epochs=opts["nepochs"],
                num_batches=opts["ds_size"] // opts["batch_size"],
            )
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))
        model = torch.load("./model.pth")
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
        default="houses_dataset.p",
        help="load the dataset if it exists, "
        "generate it and save to the path otherwise",
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

    args = parser.parse_args()
    from utils import setup

    opts = setup(args)

    main(opts)
