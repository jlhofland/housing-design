import argparse
import os
import numpy as np
import math
import json
import wandb

from housingpipeline.houseganpp.dataset.floorplan_dataset_maps_functional_high_res import (
    FloorplanGraphDataset,
    floorplan_collate_fn,
)
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps
from housingpipeline.houseganpp.misc.utils import combine_images, _init_input, selectRandomNodes, selectNodesTypes
from housingpipeline.houseganpp.models.models import Discriminator, Generator, compute_gradient_penalty

os.chdir("/home/aledbetter/housing-design/housingpipeline/housingpipeline/houseganpp/")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=100, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=2,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--sample_interval", type=int, default=1000, help="interval between image sampling"
)
parser.add_argument("--exp_folder", type=str, default="exp", help="destination folder")
parser.add_argument(
    "--n_critic",
    type=int,
    default=1,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--status_print_interval",
    type=int,
    default=500,
    help="number of batches (of size 1..) between status prints",
)
parser.add_argument(
    "--target_set",
    type=str,
    default='D',
    choices=['A', 'B', 'C', 'D', 'E'],
    help="which split to remove",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/aledbetter/datasets/hhgpp_datasets/full_datasets",
    help="path to the dataset",
)

# use checkpoint "restart" model
parser.add_argument(
    "--gen_restart_path",
    type=str,
    default=None,
    help="specify a path to a nice model to use to jumpstart training",
)
parser.add_argument(
    "--disc_restart_path",
    type=str,
    default=None,
    help="specify a path to a nice model to use to jumpstart training",
)

parser.add_argument(
    "--lambda_gp", type=int, default=10, help="lambda for gradient penalty"
)
opt = parser.parse_args()

# Initialize W and B logging
wandb.login(key="023ec30c43128f65f73c0d6ea0b0a67d361fb547")
wandb.init(project='Housing', config=vars(opt))
print(f"offline mode: {wandb.run.settings._offline}")

exp_folder = "{}_{}".format(opt.exp_folder, opt.target_set)
os.makedirs("/scratch/aledbetter/exps/" + exp_folder, exist_ok=True)

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()
distance_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator()
if opt.gen_restart_path is not None:
    generator.load_state_dict(torch.load(opt.gen_restart_path))
discriminator = Discriminator()
if opt.disc_restart_path is not None:
    discriminator.load_state_dict(torch.load(opt.disc_restart_path))
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)


# Visualize a single batch
def visualizeSingleBatch(generator, fp_loader_test, opt, exp_folder, batches_done, batch_size=8):
    print(
        "Loading saved model ... \n{}".format(
            "/scratch/aledbetter/checkpoints/gen_{}_{}.pth".format(exp_folder, batches_done)
        )
    )
    generatorTest = generator
    generatorTest.load_state_dict(
        torch.load("/scratch/aledbetter/checkpoints/gen_{}_{}.pth".format(exp_folder, batches_done))
    )
    generatorTest = generatorTest.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        generatorTest.to(device)

    with torch.no_grad():
        # Unpack batch
        mks, nds, eds, eds_f, nd_to_sample, ed_to_sample = next(iter(fp_loader_test))
        real_mks = Variable(mks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds
        given_eds_f = eds_f
        graph = [given_nds, given_eds, given_eds_f]
        # Select random nodes
        ind_fixed_nodes, _ = selectNodesTypes(nd_to_sample, batch_size, nds)
        # build input
        state = {"masks": real_mks, "fixed_nodes": ind_fixed_nodes}
        # Done: update _init_input to consider eds_f
        z, given_masks_in, given_nds, given_eds, given_eds_f = _init_input(graph, state)
        z, given_masks_in, given_nds, given_eds, given_eds_f = (
            z.to(device),
            given_masks_in.to(device),
            given_nds.to(device),
            given_eds.to(device),
            given_eds_f.to(device)
        )
        # gen_mks = generator(z, given_masks_in, given_nds, given_eds)
        # Generate a batch of images
        gen_mks = generatorTest(z, given_masks_in, given_nds, given_eds, given_eds_f)
        # Generate image tensors
        real_imgs_tensor = combine_images(
            real_mks, given_nds, given_eds, nd_to_sample, ed_to_sample
        )
        fake_imgs_tensor = combine_images(
            gen_mks, given_nds, given_eds, nd_to_sample, ed_to_sample
        )
        # Save images
        save_image(
            real_imgs_tensor,
            "/scratch/aledbetter/exps/{}/{}_real.png".format(exp_folder, batches_done),
            nrow=12,
            normalize=False,
        )
        save_image(
            fake_imgs_tensor,
            "/scratch/aledbetter/exps/{}/{}_fake.png".format(exp_folder, batches_done),
            nrow=12,
            normalize=False,
        )
        wandb.log({'Real Images': [wandb.Image("/scratch/aledbetter/exps/{}/{}_real.png".format(exp_folder, batches_done))]})
        wandb.log({'Generated Images': [wandb.Image("/scratch/aledbetter/exps/{}/{}_fake.png".format(exp_folder, batches_done))]})
        generatorTest.train()
    return


# Configure data loader
fp_dataset_train = FloorplanGraphDataset(
    opt.data_path,
    transforms.Normalize(mean=[0.5], std=[0.5]),
    target_set=opt.target_set,
)

fp_loader = torch.utils.data.DataLoader(
    fp_dataset_train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    collate_fn=floorplan_collate_fn,
    pin_memory=False,
)

fp_dataset_test = FloorplanGraphDataset(
    opt.data_path,
    transforms.Normalize(mean=[0.5], std=[0.5]),
    target_set=opt.target_set,
    split="eval",
)

fp_loader_test = torch.utils.data.DataLoader(
    fp_dataset_test,
    batch_size=8,
    shuffle=True,
    num_workers=opt.n_cpu,
    collate_fn=floorplan_collate_fn,
    pin_memory=False,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2)
)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ----------
#  Training
# ----------
batches_done = 0
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(fp_loader):
        # Unpack batch
        mks, nds, eds, eds_f, nd_to_sample, ed_to_sample= batch
        # print(f"mks:\n Shape:{mks.shape}\n {mks}")
        # print(f"nds:\n Shape:{nds.shape}\n {nds}")
        # print(f"eds:\n Shape:{eds.shape}\n {eds}")
        # print(f"nd_to_sample:\n Shape:{nd_to_sample.shape}\n {nd_to_sample}")
        # print(f"ed_to_sample:\n Shape:{ed_to_sample.shape}\n {ed_to_sample}")
        indices = nd_to_sample, ed_to_sample
        # Adversarial ground truths
        batch_size = torch.max(nd_to_sample) + 1
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_mks = Variable(mks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds
        given_eds_f = eds_f
        graph = [given_nds, given_eds, given_eds_f]
        # Set grads on
        for p in discriminator.parameters():
            p.requires_grad = True

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Select random nodes
        ind_fixed_nodes, _ = selectNodesTypes(nd_to_sample, batch_size, nds)

        # Generate a batch of images
        state = {"masks": real_mks, "fixed_nodes": ind_fixed_nodes}
        z, given_masks_in, given_nds, given_eds, given_eds_f = _init_input(graph, state)
        z, given_masks_in, given_nds, given_eds, given_eds_f = (
            z.to(device),
            given_masks_in.to(device),
            given_nds.to(device),
            given_eds.to(device),
            given_eds_f.to(device)
        )
        gen_mks = generator(z, given_masks_in, given_nds, given_eds, given_eds_f)
        # Real images
        real_validity = discriminator(real_mks, given_nds, given_eds, nd_to_sample, given_eds_f)
        # Fake images
        fake_validity = discriminator(
            gen_mks.detach(),
            given_nds.detach(),
            given_eds.detach(),
            nd_to_sample.detach(),
            given_eds_f.detach()
        )
        # Measure discriminator's ability to classify real from generated samples
        gradient_penalty = compute_gradient_penalty(
            D=discriminator,
            x=real_mks.data,
            x_fake=gen_mks.data,
            given_y=given_nds.data,
            given_w=given_eds.data,
            nd_to_sample=nd_to_sample.data,
            data_parallel=None,
            given_ed_f=given_eds_f.data,
        )
        d_loss = (
            -torch.mean(real_validity)
            + torch.mean(fake_validity)
            + opt.lambda_gp * gradient_penalty
        )
        # Update discriminator
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Set grads off
        for p in discriminator.parameters():
            p.requires_grad = False
        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:
            # Generate a batch of images

            z = Variable(
                Tensor(np.random.normal(0, 1, tuple((real_mks.shape[0], 128))))
            )
            gen_mks = generator(z, given_masks_in, given_nds, given_eds, given_eds_f)
            # Score fake images
            fake_validity = discriminator(gen_mks, given_nds, given_eds, nd_to_sample, given_eds_f)
            # Compute L1 loss
            err = (
                distance_loss(
                    gen_mks[ind_fixed_nodes, :, :],
                    given_masks_in[ind_fixed_nodes, 0, :, :],
                )
                * 1000
                if len(ind_fixed_nodes) > 0
                else torch.tensor(0.0)
            )
            # Update generator
            g_loss = -torch.mean(fake_validity) + err
            g_loss.backward()
            # Update optimizer
            optimizer_G.step()
            if (batches_done % opt.sample_interval == 0) and batches_done:
                torch.save(
                    generator.state_dict(),
                    "/scratch/aledbetter/checkpoints/gen_{}_{}.pth".format(exp_folder, batches_done),
                )
                wandb.save("/scratch/aledbetter/checkpoints/gen_{}_{}.pth".format(exp_folder, batches_done))
                torch.save(
                    discriminator.state_dict(),
                    "/scratch/aledbetter/checkpoints/disc_{}_{}.pth".format(exp_folder, batches_done),
                )
                wandb.save("/scratch/aledbetter/checkpoints/disc_{}_{}.pth".format(exp_folder, batches_done))
                visualizeSingleBatch(generator, fp_loader_test, opt, exp_folder, batches_done)
            if batches_done % opt.status_print_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [L1 loss: %f]"
                    % (
                        epoch,
                        opt.n_epochs,
                        batches_done,
                        len(fp_loader),
                        d_loss.item(),
                        g_loss.item(),
                        err.item(),
                    )
                )

            wandb.log({
            'epoch': epoch,
            'batch': batches_done,
            'D_loss': d_loss.item(),
            'G_loss': g_loss.item(),
            'L1_loss': err.item()
            })

            batches_done += opt.n_critic

wandb.finish()

"""def reader(filename):
    with open(filename) as f:
        info = json.load(f)
        rms_bbs = np.asarray(info["boxes"])
        fp_eds = info["edges"]
        rms_type = info["room_type"]
        eds_to_rms = info["ed_rm"]
        s_r = 0
        for rmk in range(len(rms_type)):
            if rms_type[rmk] != 17:
                s_r = s_r + 1
        # print("eds_ro",eds_to_rms)
        rms_bbs = np.array(rms_bbs) / 256.0
        fp_eds = np.array(fp_eds) / 256.0
        fp_eds = fp_eds[:, :4]
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        tl -= shift
        br -= shift
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):
            eds_to_rms_tmp.append([eds_to_rms[l][0]])
        return rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp
"""
