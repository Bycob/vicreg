from pathlib import Path
import functools
import numpy as np
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import augmentations as aug
from distributed import init_distributed_mode
from main_vicreg import adjust_learning_rate, Projector, exclude_bias_and_norm, LARS, batch_all_gather, FullGatherLayer
from resnet_encoder import ResnetEncoder



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain an encoder with VICReg", add_help=False)

    parser.add_argument("--data", type=str, default="dogs-cats", choices=("dogs-cats", "airbus-cracks"), help="chosen dataset")
    parser.add_argument("--exp-dir", type=Path, default="./exp", help="Path to the experiment folder, where all logs/checkpoints will be stored")
    parser.add_argument("--log-freq-time", type=int, default=60, help="Print logs to the stats.txt file every [log-freq-time] seconds")
    parser.add_argument("--mlp", default="8192-8192-8192", help="Size and number of layers of the MLP expander head")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Effective batch size (per worker batch size is [batch-size] / world-size")
    parser.add_argument("--base-lr", type=float, default=0.2, help="Base learning rate, effective learning rate after warmup is [base-lr] * [batch-size] / 256")
    parser.add_argument("--wd", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--sim-coeff", type=float, default=25.0, help="Invariance regularization loss coefficient")
    parser.add_argument("--std-coeff", type=float, default=25.0, help="Variance regularization loss coefficient")
    parser.add_argument("--cov-coeff", type=float, default=1.0, help="Covariance regularization loss coefficient")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", default="cuda", help="Device to use for training / testing")
    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    return parser




def main(args):
    
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

        
    transforms = aug.TrainTransform()

    if args.data == "dogs-cats":
        dataset = datasets.ImageFolder("/data1/jeanne/datasets/dogs_cats", transforms)

    elif args.data == "airbus-cracks":
        dataset = datasets.ImageFolder("/data1/jeanne/datasets/airbus_cracks_crops", transforms)

    else:
        assert False

   
    assert args.batch_size % args.world_size == 0
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(dataset / "train", **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset / "test", **kwargs)
    

    model = Encoder_VICReg(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.wd, weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)

    
    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0



    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            print(x.shape())
            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(epoch=epoch, step=step, loss=loss.item(), time=int(current_time - start_time), lr=lr)
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        if args.rank == 0:
            state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")


class Encoder_VICReg(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone = ResnetEncoder(input_nc=3, output_nc=3)
        self.projector = Projector(args, self.embedding'mets juste le nombre de trucs en sortie à la place')

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (self.args.sim_coeff * repr_loss + self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss)

        return loss


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
